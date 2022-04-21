import csv

# Source: https://realpython.com/python-csv/
def read_csv(filename):
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = []
        line_count = 0
        for row in csv_reader:
        #     if line_count == 0:
        #         print(f'Column names are {", ".join(row)}')
        #         line_count += 1
            rows.append(row)
        #     line_count += 1
        # print(f'Processed {line_count} lines.')
        return rows

def get_coord_cities(list_dict):
    cities = {}
    for e in list_dict:
        cities[e["city"]] = [e["geo_lat"], e["geo_lon"]]
    return cities