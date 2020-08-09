import time


def progress(current_item, total_items):
    current_item = current_item + 1

    complete = current_item / total_items * 100
    num_hashes = int(complete / 10)
    num_dots = 10 - num_hashes

    hashes = '#' * num_hashes
    dots = '.' * num_dots
    progress_str = f'Progress: [ {hashes}{dots} ] {round(complete, 2)}%'

    if (current_item < total_items):
        endchar = '\r'
    else:
        endchar = '\n'

    print(progress_str, sep='', end=endchar)


if __name__ == '__main__':
    max_num = 700
    for i in range(max_num):
        time.sleep(0.01)
        progress(i + 1, max_num)
