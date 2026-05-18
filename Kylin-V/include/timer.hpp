#pragma once
#include "util.hpp"
struct Timer
{
    map<string_view,D> cpu_times;
    map<string_view,D> wall_times;
    clock_t cpu_start;
    clock_t cpu_end;
    std::chrono::time_point<std::chrono::high_resolution_clock> wall_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> wall_end;
    void start_module(const char * m)
    {
        string_view module(m);
        if(cpu_times.find(module) == cpu_times.end())
        {
            cpu_times[module] = 0.0;
            wall_times[module] = 0.0;
        }
        cpu_start = clock();
        wall_start = std::chrono::high_resolution_clock::now();
    }
    void end_module(const char * m)
    {
        string_view module(m);
        if(cpu_times.find(module) == cpu_times.end())
        {
            cout << "No module named " << module << "!" << endl;
            exit(1);
        }
        cpu_end = clock();
        wall_end = std::chrono::high_resolution_clock::now();
        cpu_times[module] += 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
        wall_times[module] += std::chrono::duration<D,std::milli>(wall_end-wall_start).count();
    }
};
ostream & operator<<(ostream & os, Timer & tm)
{
    os << std::left << setw(15) << "Modules"
    << setw(15) << "Cpu times/ms"
    << setw(15) << "Wall times/ms"
    << setw(15) << "Percentage" << endl;
    
    D TotalWall = 0.0;
    for(auto it=tm.wall_times.begin();it!=tm.wall_times.end();++it)
    {
        TotalWall += it->second;
    }
    for(auto it=tm.wall_times.begin();it!=tm.wall_times.end();++it)
    {
        os << std::left << setw(15) << it->first << setw(15) << setprecision(2) << std::scientific << tm.cpu_times[it->first]
        << setw(15) << setprecision(2) << std::scientific << tm.wall_times[it->first]
        << setw(15) << setprecision(2) << std::fixed << tm.wall_times[it->first] / TotalWall * 100.0
        << endl;
    }
    return os;
}
namespace GlobalTimer {
    Timer timer;
}
