Задание: реализовать сложение двух векторов.
1. task1_1.cu - *<1, N>* и *<N, 1>* конфигурации ядра
2. task1_2.cu - *<<<(Num_elements+(Num_threads-1))/Num_threads, Num_threads>>>* конфигурация ядра
3. task1_enchanced.cu - улучшенная версия алгоритма, базируется версии, представленной в книге **Jason Senders. CUDA by Example. First printing.** (глава 5.2.1)