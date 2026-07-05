// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#include "BackendRegistry.h"

BackendRegistry& BackendRegistry::instance()
{
    static BackendRegistry registry;
    return registry;
}

void BackendRegistry::registerBackend(const std::string& name, Factory factory)
{
    if (factories.find(name) == factories.end())
        order.push_back(name);
    factories[name] = std::move(factory);
}

std::unique_ptr<InferenceBackend> BackendRegistry::create(const std::string& name) const
{
    auto it = factories.find(name);
    if (it == factories.end())
        return nullptr;
    return it->second();
}

bool BackendRegistry::has(const std::string& name) const
{
    return factories.find(name) != factories.end();
}
