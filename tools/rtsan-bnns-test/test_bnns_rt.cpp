// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
// API-level RealtimeSanitizer probe for BNNSGraphContextExecute.

#include <Accelerate/Accelerate.h>
#include <mach/mach.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

[[noreturn]] void die(const char* message)
{
    fprintf(stderr, "FATAL: %s\n", message);
    exit(1);
}

struct BNNSTestState
{
    bnns_graph_t graph{};
    bnns_graph_context_t context{};
    char* workspace = nullptr;
    size_t workspaceSize = 0;
    size_t inputIndex = 0;
    size_t outputIndex = 0;
    size_t argumentCount = 0;
    float* input = nullptr;
    float* output = nullptr;
    size_t inputBytes = 0;
    size_t outputBytes = 0;
};

size_t pageAligned(size_t bytes)
{
    const size_t pageSize = vm_page_size;
    return ((bytes + pageSize) + pageSize - 1) & ~(pageSize - 1);
}

BNNSTestState setup(const char* modelPath)
{
    BNNSTestState state;
    auto options = BNNSGraphCompileOptionsMakeDefault();
    BNNSGraphCompileOptionsSetTargetSingleThread(options, true);
    state.graph = BNNSGraphCompileFromFile(modelPath, nullptr, options);
    BNNSGraphCompileOptionsDestroy(options);
    if (state.graph.data == nullptr)
        die("BNNSGraphCompileFromFile failed");

    state.context = BNNSGraphContextMake(state.graph);
    if (state.context.data == nullptr)
        die("BNNSGraphContextMake failed");
    if (BNNSGraphContextSetArgumentType(
            state.context, BNNSGraphArgumentTypePointer) != 0)
        die("BNNSGraphContextSetArgumentType failed");

    state.inputIndex = BNNSGraphGetArgumentPosition(state.graph, nullptr, "input");
    state.outputIndex = BNNSGraphGetArgumentPosition(state.graph, nullptr, "output");
    state.argumentCount = BNNSGraphGetArgumentCount(state.graph, nullptr);
    if (state.inputIndex == SIZE_MAX || state.outputIndex == SIZE_MAX ||
        state.argumentCount == SIZE_MAX || state.argumentCount > 16)
        die("failed to discover graph arguments");

    const size_t rawWorkspace = BNNSGraphContextGetWorkspaceSize(state.context, nullptr);
    if (rawWorkspace == SIZE_MAX)
        die("BNNSGraphContextGetWorkspaceSize failed");
    state.workspaceSize = pageAligned(rawWorkspace);
    state.workspace = static_cast<char*>(aligned_alloc(vm_page_size, state.workspaceSize));
    if (state.workspace == nullptr)
        die("workspace allocation failed");
    memset(state.workspace, 0, state.workspaceSize);

    state.inputBytes = 64 * sizeof(float);
    state.outputBytes = 64 * sizeof(float);
    state.input = static_cast<float*>(malloc(state.inputBytes));
    state.output = static_cast<float*>(malloc(state.outputBytes));
    if (state.input == nullptr || state.output == nullptr)
        die("I/O allocation failed");
    for (int i = 0; i < 64; ++i)
        state.input[i] = static_cast<float>(i) / 64.0f;
    return state;
}

void audioCallback(BNNSTestState& state) [[clang::nonblocking]]
{
    bnns_graph_argument_t arguments[16] = {};
    arguments[state.inputIndex] = {
        .data_ptr = state.input,
        .data_ptr_size = state.inputBytes,
    };
    arguments[state.outputIndex] = {
        .data_ptr = state.output,
        .data_ptr_size = state.outputBytes,
    };
    if (BNNSGraphContextExecute(
            state.context, nullptr, state.argumentCount, arguments,
            state.workspaceSize, state.workspace) != 0)
        __builtin_trap();
}

void teardown(BNNSTestState& state)
{
    free(state.input);
    free(state.output);
    free(state.workspace);
    BNNSGraphContextDestroy(state.context);
    free(state.graph.data);
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <path/to/SimpleAudioMLP.mlmodelc>\n", argv[0]);
        return 2;
    }

    auto state = setup(argv[1]);
    for (int i = 0; i < 100; ++i)
    {
        state.input[0] = static_cast<float>(i) / 100.0f;
        audioCallback(state);
    }
    printf("PASS: 100 BNNSGraphContextExecute calls produced no RTSan violation\n");
    teardown(state);
    return 0;
}
