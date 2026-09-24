// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Repeated CPU-only whisper.cpp PCM-to-text benchmark. Build against the pinned
// reference and supply the verified FP32 ggml model, raw JFK PCM, thread count,
// and timed iteration count. Five warm calls are excluded from stdout timings.
#include "whisper.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static std::string trim(std::string s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    const auto end = s.find_last_not_of(" \t\r\n");
    return begin == std::string::npos ? "" : s.substr(begin, end - begin + 1);
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s model pcm-f32le threads iterations\n", argv[0]);
        return 2;
    }
    const int threads = std::atoi(argv[3]);
    const int iterations = std::atoi(argv[4]);
    if (threads < 1 || iterations < 1) return 2;
    std::ifstream file(argv[2], std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (bytes.empty() || bytes.size() % sizeof(float)) {
        std::fprintf(stderr, "invalid PCM input\n");
        return 2;
    }
    std::vector<float> pcm(bytes.size() / sizeof(float));
    std::copy(bytes.begin(), bytes.end(), reinterpret_cast<char *>(pcm.data()));
    auto cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    auto * ctx = whisper_init_from_file_with_params(argv[1], cparams);
    if (!ctx) return 2;
    auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = threads;
    params.language = "en";
    params.no_timestamps = true;
    params.temperature = 0.0f;
    params.temperature_inc = 0.0f;
    params.greedy.best_of = 1;
    params.beam_search.beam_size = 1;
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    const std::string expected = "And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.";
    for (int i = -5; i < iterations; ++i) {
        whisper_reset_timings(ctx);
        auto start = std::chrono::steady_clock::now();
        const int rc = whisper_full(ctx, params, pcm.data(), static_cast<int>(pcm.size()));
        auto end = std::chrono::steady_clock::now();
        if (rc) return 3;
        std::string text;
        for (int j = 0; j < whisper_full_n_segments(ctx); ++j) {
            text += whisper_full_get_segment_text(ctx, j);
        }
        if (trim(text) != expected) {
            std::fprintf(stderr, "transcript mismatch: %s\n", text.c_str());
            return 4;
        }
        if (i >= 0) {
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            std::printf("%lld\n", static_cast<long long>(ns));
            std::fprintf(stderr,"SINGLECORE_SAMPLE %d\n",i);
            whisper_print_timings(ctx);
            std::vector<int> ids;
            for(int segment=0;segment<whisper_full_n_segments(ctx);++segment) {
                for(int token=0;token<whisper_full_n_tokens(ctx,segment);++token) {
                    int id=whisper_full_get_token_id(ctx,segment,token);
                    if(i==0) std::fprintf(stderr,"TOKEN %d\n",id);
                    if(id<50256) ids.push_back(id);
                }
            }
            const std::vector<int> expectedIds={843,523,616,5891,3399,1265,407,644,534,1499,460,466,329,345,11,1265,644,345,460,466,329,534,1499,13};
            if(ids!=expectedIds) {std::fprintf(stderr,"token mismatch\n");return 5;}
        }
    }
    whisper_free(ctx);
    return 0;
}
