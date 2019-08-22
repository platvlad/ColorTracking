#include "utils/OptProcessor.hpp"

#include <iostream>

namespace utils
{

    OptProcessor::OptProcessor():
        m_desc("Options")
    {
        addOptions()
                ("help,h", "Print usage");
    }

    boost::program_options::options_description_easy_init 
            OptProcessor::addOptions()
    {
        return m_desc.add_options();
    }

    OptProcessor::Result OptProcessor::process(int argc, char *argv[])
    {
        using namespace boost::program_options;

        try {
            variables_map vm;
            store(parse_command_line(argc, argv, m_desc), vm);

            if (vm.count("help")) {
                std::cout << m_desc << std::endl;
                Result const result = {true, 0};
                return result;
            }
            notify(vm);
        } catch (std::exception &e) {
            std::cerr << "Argument parsing error: " << e.what() << "\n";
            Result const result = {true, -2};
            return result;
        }

        Result const result = {false, 0};
        return result;
    }

}
// namespace utils
