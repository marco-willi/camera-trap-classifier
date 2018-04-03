# import unittest
# from data_processing.data_importer import ImportFromPantheraCSV
#
#
#
#
#
# """ Class To Import and Read Datasets """
# import os
# import json
# import csv
#
# from config.config import logging
# from data_processing.utils import clean_input_path
#
#
# path_to_csv = '.\\test\\test_files\\inventory_list_example.csv'
# os.path.exists(path_to_csv)
# data_dict = dict()
# with open(path_to_csv, 'r') as f:
#     csv_reader = csv.reader(f, delimiter=',', quotechar='"')
#     for i, row in enumerate(csv_reader):
#         # check header
#         if i ==0:
#             assert row == ['image', 'species', 'count', 'survey', 'dir']
#         else:
#             # extract fields from csv
#             _id = row[0]
#             species = row[1]
#             try:
#                 species_count = int(row[2])
#             except:
#                 if row[2] == 'NA':
#                     species_count = -1
#                 else:
#                     print("Record: %s has invalid count: %s" % (_id, species_count))
#             survey = row[3]
#             image_path = row[4]
#             if _id in data_dict:
#                 print("ID: %s already exists - skipping" % _id)
#             else:
#                 data_dict[_id] = {
#                         'images': [image_path],
#                         'labels': {'species': [species],
#                                    'counts': [species_count]}
#                         }
#
#
# from data_processing.data_importer import ImportFromPantheraCSV
#
#
# csv_importer = ImportFromPantheraCSV()
# data_dict = csv_importer.read_from_csv(path_to_csv)
