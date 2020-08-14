#!/usr/bin/env python3
#encoding=utf8
import os
import re
import sys
import sqlite3
from collections import Counter

from tqdm import tqdm

def file_lines(file_path):
    with open(file_path, 'rb') as fp:
        b = fp.read()
    content = b.decode('utf8', 'ignore')
    lines = []
    for line in tqdm(content.split('\n')):
        try:
            line = line.replace('\n', '').strip()
            if line.startswith('E'):
                lines.append('')
            elif line.startswith('M '):
                chars = line[2:].split('/')
                while len(chars) and chars[len(chars) - 1] == '.':
                    chars.pop()
                if chars:
                    sentence = ''.join(chars)
                    sentence = re.sub('\s+', '，', sentence)
                    lines.append(sentence)
        except:
            print(line)
            return lines
        
        lines.append('')
    return lines

def contain_chinese(s):
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False

def valid(a, max_len=0):
    if len(a) > 0 and contain_chinese(a):
        if max_len <= 0:
            return True
        elif len(a) <= max_len:
            return True
    return False

def insert(a, b, cur):
    cur.execute("""
    INSERT INTO conversation (ask, answer) VALUES
    ('{}', '{}')
    """.format(a.replace("'", "''"), b.replace("'", "''")))

def insert_if(question, answer, cur, input_len=500, output_len=500):
    if valid(question, input_len) and valid(answer, output_len):
        insert(question, answer, cur)
        return 1
    return 0

def main(file_path):
    lines = file_lines(file_path)

    print('一共读取 %d 行数据' % len(lines))

    db = 'db/conversation.db'
    if os.path.exists(db):
        os.remove(db)
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation
        (ask text, answer text);
        """)
    conn.commit()

    words = Counter()
    a = ''
    b = ''
    inserted = 0

    for index, line in tqdm(enumerate(lines), total=len(lines)):
        words.update(Counter(line))
        a = b
        b = line
        ask = a
        answer = b
        inserted += insert_if(ask, answer, cur)
        # 批量提交
        if inserted != 0 and inserted % 50000 == 0:
            conn.commit()
    conn.commit()

if __name__ == '__main__':
    file_path = 'dgk_shooter_min.conv'
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print('文件 {} 不存在'.format(file_path))
    else:
        main(file_path)
