#!/bin/bash

# Имя исходного файла
SOURCE_FILE="for.c"
# Имя выходного файла для обычной компиляции
OUTPUT_NORMAL="program_normal"
# Имя выходного файла для компиляции с OpenMP
OUTPUT_OPENMP="program_openmp"
# Файл для записи результатов
RESULT_FILE="itog.txt"

# Очистка файла с результатами
> "$RESULT_FILE"

# Компиляция без OpenMP
echo "Компиляция без OpenMP..."
gcc "$SOURCE_FILE" -o "$OUTPUT_NORMAL"
if [ $? -ne 0 ]; then
    echo "Ошибка при компиляции программы без OpenMP."
    exit 1
fi

echo "Запуск программы без OpenMP..." >> "$RESULT_FILE"
./"$OUTPUT_NORMAL" >> "$RESULT_FILE"

# Компиляция с OpenMP
echo "Компиляция с OpenMP..."
gcc "$SOURCE_FILE" -o "$OUTPUT_OPENMP" -fopenmp
if [ $? -ne 0 ]; then
    echo "Ошибка при компиляции программы с OpenMP."
    exit 1
fi

# Запуск программы с OpenMP на 1, 2, 4, 8 потоках
for THREADS in 1 2 4 8 16 28; do
    echo "Запуск программы с OpenMP на $THREADS потоках..." >> "$RESULT_FILE"
    export OMP_NUM_THREADS=$THREADS
    ./"$OUTPUT_OPENMP" >> "$RESULT_FILE"
done

# Уведомление об успешном завершении
echo "Выполнение завершено. Результаты записаны в $RESULT_FILE."
