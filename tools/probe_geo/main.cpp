// probe_geo — hidden-state export for the section-probe wind tunnel.
// Tokenizes each section file, slices fixed 512-token chunks, samples
// N_PER_SECTION per section (seeded RNG), decodes each sampled chunk in
// one llama_decode call, and lets the STQ_RESIDUAL_LOG hook capture
// per-layer residual stats (mean-pool + final token) per eval.
//
// usage: probe_geo -m model.gguf -s sections.txt -o outdir -n 200
//   sections.txt: one "name<TAB>path" per line (code/prose/disc/commits)
// outputs:
//   outdir/contexts.json : [{section, chunk, n_tokens}, ...] in eval order
//   outdir/resid.bin     : written by the hook (2*dim f32 per layer per eval)
#include "llama.h"
#include "common.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>

struct Section { std::string name, path; };

int main(int argc, char ** argv) {
    const char * model = NULL, * secfile = NULL, * outdir = NULL;
    int n_per = 200, seed = 1337;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m")) model = argv[++i];
        else if (!strcmp(argv[i], "-s")) secfile = argv[++i];
        else if (!strcmp(argv[i], "-o")) outdir = argv[++i];
        else if (!strcmp(argv[i], "-n")) n_per = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")) seed = atoi(argv[++i]);
    }
    if (!model || !secfile || !outdir) {
        fprintf(stderr, "usage: probe_geo -m model.gguf -s sections.txt -o outdir [-n 200] [--seed 1337]\n");
        return 1;
    }
    std::vector<Section> sections;
    std::ifstream sf(secfile);
    std::string line;
    while (std::getline(sf, line)) {
        if (line.empty()) continue;
        auto t = line.find('\t');
        if (t == std::string::npos) continue;
        sections.push_back({line.substr(0, t), line.substr(t + 1)});
    }
    if (sections.size() != 4) { fprintf(stderr, "need 4 sections, got %zu\n", sections.size()); return 1; }

    std::string cmd = "mkdir -p " + std::string(outdir);
    if (system(cmd.c_str())) return 1;
    std::string respath = std::string(outdir) + "/resid.bin";
    setenv("STQ_RESIDUAL_LOG", respath.c_str(), 1);
    setenv("STQ_NO_FUSE", "1", 1);  // keep attn_norm nodes un-fused for capture

    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    llama_model * lmodel = llama_model_load_from_file(model, mp);
    if (!lmodel) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(lmodel);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 640;
    cp.n_batch = 640;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(lmodel, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    const int n_ctx = 512;
    std::mt19937 rng(seed);
    std::vector<std::string> order_names;
    std::vector<int> order_chunks, order_ntok;
    long total_evals = 0;

    for (auto & sec : sections) {
        std::ifstream in(sec.path, std::ios::binary);
        std::stringstream buf; buf << in.rdbuf();
        std::string text = buf.str();
        std::vector<llama_token> toks(text.size() / 2 + 8);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), (int) toks.size(), false, true);
        if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), (int) toks.size(), false, true); }
        fprintf(stderr, "[%s] %d tokens\n", sec.name.c_str(), n);
        int n_chunks = n / n_ctx;
        if (n_chunks < n_per) { fprintf(stderr, "[%s] only %d chunks < %d\n", sec.name.c_str(), n_chunks, n_per); return 1; }
        // stratified even spread, seeded jitter: pick n_per of n_chunks
        std::vector<int> picks;
        for (int i = 0; i < n_per; ++i) {
            double base = (double) i * n_chunks / n_per;
            int j = (int) base + (int) (rng() % (int) ((double) (i + 1) * n_chunks / n_per - base + 1));
            picks.push_back(j < n_chunks ? j : n_chunks - 1);
        }
        for (int j : picks) {
            llama_batch b = llama_batch_init(n_ctx, 0, 0);
            for (int t = 0; t < n_ctx; ++t) {
                b.token[t] = toks[j * n_ctx + t];
                b.pos[t] = t;
                b.n_seq_id[t] = 1;
                b.seq_id[t][0] = 0;
                b.logits[t] = (t == n_ctx - 1);
            }
            b.n_tokens = n_ctx;
            if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed %s#%d\n", sec.name.c_str(), j); return 1; }
            llama_batch_free(b);
            llama_memory_clear(llama_get_memory(ctx), true);
            order_names.push_back(sec.name);
            order_chunks.push_back(j);
            order_ntok.push_back(n_ctx);
            total_evals++;
            if (total_evals % 50 == 0) fprintf(stderr, "  %ld evals done\n", total_evals);
        }
    }
    std::ofstream cj(std::string(outdir) + "/contexts.json.tmp");
    cj << "[\n";
    for (size_t i = 0; i < order_names.size(); ++i) {
        cj << "{\"section\": \"" << order_names[i] << "\", \"chunk\": " << order_chunks[i]
           << ", \"n_tokens\": " << order_ntok[i] << "}" << (i + 1 < order_names.size() ? "," : "") << "\n";
    }
    cj << "]\n";
    cj.close();
    std::rename((std::string(outdir) + "/contexts.json.tmp").c_str(), (std::string(outdir) + "/contexts.json").c_str());
    fprintf(stderr, "done: %ld evals -> %s\n", total_evals, respath.c_str());
    return 0;
}
