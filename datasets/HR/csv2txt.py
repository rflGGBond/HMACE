import csv
from tqdm import tqdm

if __name__ == '__main__':
    input_file = "HR_edges.csv"
    output_file = "HR.txt"

    with open(input_file, 'r', encoding='utf-8') as csv_file, \
        open(output_file, 'w', encoding='utf-8') as txt_file:
        reader = csv.reader(csv_file)
        for row in tqdm(reader):
            node1, node2 = row
            txt_file.write(f"{node1} {node2}\n")

    print(f"✅ 已将 {input_file} 转换为 {output_file}")