#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>


#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  100
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;

    struct llama_sampling_context * ctx_sampling;
};

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    
    if (params.model_draft.empty()) {
        fprintf(stderr, "%s: error: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for accepting a token from the draft model
    const float p_accept = params.p_accept;

    // probability threshold for splitting a draft branch (only for 1 > 1)
    const float p_split  = params.p_split;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("l2", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    std::vector<int> static_draft = {330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723, 13};
    std::vector<int> random_draft(100, 330);
    // std::vector<int> static_draft = {330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723, 13};
    std::vector<int> full_prompt_draft = {1, 733, 16289, 28793, 1298, 15882, 272, 14804, 11645, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 611, 28705, 28740, 28734, 2421, 733, 28748, 16289, 28793, 387, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723, 13, 28733, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723, 13, 28733, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723, 13, 28733, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723, 13, 28733, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723, 13, 28733, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723, 13, 28733, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723, 13, 28733, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723, 13, 28733, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723, 13, 28733, 330, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 9540, 28723};

    // LOG the static draft using LOG_TEE
    LOG_TEE("Static draft: ");
    // LOG_TEE the static draft using LOG_TEE
    for (auto id : static_draft) {
        LOG_TEE("%d", id);
    }
    LOG_TEE("\n");

    // init llama.cpp
    llama_backend_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    params.logits_all = true;
    std::tie(model_tgt, ctx_tgt) = llama_init_from_gpt_params(params);


    // Tokenize the prompt
    const bool add_bos_tgt = llama_should_add_bos_token(model_tgt);
    LOG("add_bos tgt: %d\n", add_bos_tgt);


    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx_tgt, params.prompt, add_bos_tgt, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    fprintf(stderr, "\n\n");

    for (auto id : inp) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx_tgt, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1, 0,           0));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1, n_input - 1, 0));

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_n_vocab(model_dft));

    // how many tokens to draft each time
    int n_draft = params.n_draft;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    // draft sequence data
    seq_draft draft;

    params.sparams.grammar.clear(); // the draft samplers will copy the target sampler's grammar
    params.sparams.temp = -1.0f;    // force greedy sampling with probs for the draft model
    

    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    draft.i_batch_tgt.resize(1);
    draft.i_batch_tgt[0] = 0;

    long long tot_time = 0;
    long num_calls = 0;

    while (true) {
        // print current draft sequences

        const auto & tokens = draft.tokens;
        LOG("draft %d: %s\n", 0, LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, tokens).c_str());
        
        int i_dft  = 0;

        while (true) {
            LOG("sampling target: 0 = %3d, i_dft = %3d, i_batch_tgt = %3d\n", 0, i_dft, draft.i_batch_tgt[i_dft]);

            // sample from the target model
            llama_token id = llama_sampling_sample(ctx_sampling, ctx_tgt, NULL, draft.i_batch_tgt[i_dft]);

            llama_sampling_accept(ctx_sampling, ctx_tgt, id, true);


            const std::string token_str = llama_token_to_piece(ctx_tgt, id);

            if (!params.use_color) {
                printf("%s", token_str.c_str());
            }

            if (id == llama_token_eos(model_tgt)) {
                has_eos = true;
            }

            ++n_predict;

            // check if the target token matches any of the drafts
            {
                bool matches = false;

                if (i_dft < (int) draft.tokens.size() && id == draft.tokens[i_dft]) {
                    LOG("the sampled target token matches the %dth drafted token of sequence 0 (%d, '%s') - accepted\n", i_dft, id, token_str.c_str());
                    matches = true;
                }                 

                if (matches) {
                    ++n_accept;
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;
                    if (params.use_color) {
                        // Color token according to its origin sequence
                        printf("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
                        fflush(stdout);
                    }
                    continue;
                }
            }
            if (params.use_color) {
                printf("%s", token_str.c_str());
            }
            fflush(stdout);

            LOG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

            // TODO: simplify
            {
                LOG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", 0, n_past_tgt, n_past_dft);

                llama_kv_cache_seq_rm  (ctx_tgt, 0, n_past_tgt, -1);
                llama_kv_cache_seq_keep(ctx_tgt, 0);
                llama_kv_cache_seq_cp  (ctx_tgt, 0, 0, -1, -1);
                llama_kv_cache_seq_keep(ctx_tgt, 0);
            }

            
            draft.active = false;
            draft.tokens.clear();
            draft.i_batch_tgt.clear();
            
            // note: will be erased after the speculation phase
            draft.tokens.push_back(id);
            draft.i_batch_tgt.push_back(0);

            ++n_past_dft;

            break;
        }

        if (n_predict > params.n_predict || has_eos) {
            break;
        }


        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        draft.active      = true;
        draft.drafting    = true;
        draft.i_batch_dft = 0;

        llama_batch_clear(batch_tgt);
        llama_batch_add  (batch_tgt, draft.tokens[0], n_past_tgt, { 0 }, true);


        // GETTING DRAFT TOKENS

        // auto prompt_lookup = [&]() -> void {
        //     int inp_size = inp.size();
        //     for (int ngram_size = 3 ; ngram_size > 0; --ngram_size){
        //         const llama_token * ngram = &inp[inp_size - ngram_size];

        //         for (int i = 0; i <= (int) inp_size - (ngram_size * 2); ++i) {
        //             bool match = true;
        //             for (int j = 0; j < ngram_size; ++j) {
        //                 if (inp[i + j] != ngram[j]) {
        //                     match = false;
        //                     break;
        //                 }
        //             }

        //             if (match) {
        //                 const int startIdx = i + ngram_size;
        //                 const int endIdx = startIdx + n_draft;
        //                 if (endIdx < inp_size) {
        //                     for (int j = startIdx, count=0; j < endIdx; ++j, ++count) {
        //                         const llama_token id = inp[j];
        //                         LOG(" - draft candidate %d: %d\n", j, id);
        //                         draft.tokens.push_back(id);
        //                         draft.i_batch_tgt.push_back(batch_tgt.n_tokens);
        //                         llama_batch_add(batch_tgt, id, n_past_tgt + count + 1, { 0 }, true);
        //                         // llama_batch_add(batch_tgt, inp[j], n_past + (j - startIdx) + 1, { 0 }, true);
        //                         ++n_drafted;
        //                         ++n_past_cur;
        //                     }
        //                     return;
        //                 }
        //             }
        //         }
        //     }
        //     return;
        // };

        // prompt_lookup();
        // sample n_draft tokens from the draft model using prompt lookup
        for (int i = 0; i < n_draft; ++i) {
            draft.skip = false;
           
            // const llama_token id = random_draft[i];
            const llama_token id = full_prompt_draft[i+n_past_tgt+1];
            draft.tokens.push_back(id);

            // add unique drafted tokens to the target batch
            draft.i_batch_tgt.push_back(batch_tgt.n_tokens);

            llama_batch_add(batch_tgt, id, n_past_tgt + i + 1, { 0 }, true);
            
            
            ++n_past_cur;
            ++n_drafted;
        }

        // evaluate the target model on the drafted tokens
        {
            llama_kv_cache_seq_keep(ctx_tgt, 0);
            llama_kv_cache_seq_cp(ctx_tgt, 0, 0, -1, -1);
            
            LOG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            
            auto start = std::chrono::high_resolution_clock::now();

            llama_decode(ctx_tgt, batch_tgt);

            // Get the time after the operation
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate the difference, which is the time taken by the operation
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            // Add the time taken to tot_time
            tot_time += duration;
            num_calls += 1;

            ++n_past_tgt;
        }

        // the first token is always proposed by the target model before the speculation loop so we erase it here
        draft.tokens.erase(draft.tokens.begin());
        
    }

    auto t_dec_end = ggml_time_us();

    LOG_TEE("\n\n");

    // LOG tot_time
    LOG_TEE("tot_time: %lld\n", tot_time);
    // LOG num_calls
    LOG_TEE("num_calls: %ld\n", num_calls);
    // LOG float val of tot_time / num_calls
    LOG_TEE("avg_time: %f\n", (float) tot_time / num_calls);
    
    LOG_TEE("\n\n");


    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_TEE("\n");
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("n_drafted = %d\n", n_drafted);
    LOG_TEE("n_accept  = %d\n", n_accept);
    LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);


    LOG_TEE("\ntarget:\n");
    llama_print_timings(ctx_tgt);

    LOG_TEE("\nHERE:\n");
    llama_sampling_free(ctx_sampling);

    llama_sampling_free(draft.ctx_sampling);
   


    llama_free(ctx_tgt);
    llama_free_model(model_tgt);


    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
