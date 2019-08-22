#ifndef UTILS_OPT_PROCESSOR_HPP
#define UTILS_OPT_PROCESSOR_HPP

#include <boost/program_options.hpp>
#include <boost/optional.hpp>
#include <boost/any.hpp>

namespace utils
{
    class OptProcessor
    {
    public:
        OptProcessor();

        boost::program_options::options_description_easy_init addOptions();

        struct Result
        {
            bool shutdown;
            int returnCode;
        };

        Result process(int argc, char *argv[]);

    private:
        boost::program_options::options_description m_desc;
    };

}
// namespace utils


namespace boost
{
    // support for optional in program options
    // https://svn.boost.org/trac10/ticket/7495
    template<class T>
    void validate(boost::any& v, std::vector<std::string> const& values,
            boost::optional<T>* typeTag, int) {
        if (!values.empty()) {
            boost::any a;
            using namespace boost::program_options;
            validate(a, values, (T*)0, 0);
            v = boost::any(boost::optional<T>(boost::any_cast<T>(a)));
        }
    }

}
// namespace boost


#endif // UTILS_OPT_PROCESSOR_HPP
