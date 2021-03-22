#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <ctime>
#include <chrono>


int min (int a, int b){
    return a < b ? a : b;
}


class File {
public:
    std::string text;
    int length;

    File() {
        this->text = "";
        this->length = 0;
    }

    void read(std::string filename) {
        std::ifstream input(filename);
        std::string line;

        if (input.is_open()) {
            while (getline(input, line)){
                this->text.append(line + '\n');
                this->length += line.length() + 1;
            }
        }
        input.close();
    }

    int count_sequences(std::string &seq) {
        int count = 0;
        int max_threads = omp_get_max_threads();
        int size_part = this->length/max_threads;
        int remain_part = this->length % max_threads;

        #pragma omp parallel for reduction(+ : count)
        for (int i = 0; i < max_threads; i++) {
            int left = i*size_part;
            int right = left + size_part;
            if (i == max_threads - 1) {
                right += remain_part;
            }

            count += search_sequence(seq, left, right).size();
        }

        return count;
    }

    // search sequances in text within borders [left, right)
    // return absolute indexes of the first symbol for each found sequence
    std::vector<int> search_sequence(std::string &seq, int left=0, int right=-1) {
        std::vector<int> results;
        int padding;
        bool is_found;

        if (right == -1) {
            right = this->length;
        }

        for (int i = left; i < min(right, this->length - seq.length()); i++) {
            if (this->text[i] == '\n') {
                continue;
            }

            is_found = true;
            padding = 0;

            for (int j = 0; j < seq.length(); j++) {
                if (seq[j] != this->text[i + j]) {
                    if (this->text[i + j] == '-') {
                        if (this->text[i + j + 1] == '\n') {
                            j -= 1;
                            i += 2;
                            padding += 2;
                            continue;
                        }
                    }
                    else if (this->text[i + j] == '\n') {
                        j -= 1;
                        i += 1;
                        padding += 1;
                        continue;
                    }

                    is_found = false;
                    break;
                }
            }
            
            i -= padding;
            if (is_found) {
                results.push_back(i);
            }
        }

        return results;
    }
};


int main() {
    File f;
    std::string filename;
    std::string keyword;
    int res;

    std::getline(std::cin, filename);
    std::getline(std::cin, keyword);

    std::clock_t c_start = std::clock();
    auto t_start = std::chrono::high_resolution_clock::now();

    f.read(filename);
    res = f.count_sequences(keyword);

    std::clock_t c_end = std::clock();
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << std::fixed << std::setprecision(8);
    std::cout << std::endl;
    std::cout << "---text: " << std::endl << f.text << std::endl;
    std::cout << "---count word " << keyword << ": " << res << std::endl;
    std::cout << "---CPU time used: " << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "---Wall clock time passed: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;

    return 0;
}
