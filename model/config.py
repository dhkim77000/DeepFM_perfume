item_table_path = '/home/dhkim/Fragrance/data/DB.csv'
rating_table_path = '/home/dhkim/Fragrance/data/rating_table.csv'
notes_info_path = '/home/dhkim/Fragrance/data/notes_group.csv'
user_info_path = '/home/dhkim/Fragrance/data/user_info.csv'
user_dummy_path = '/home/dhkim/Fragrance/data/user_dummy.csv'
item_dummy_path = '/home/dhkim/Fragrance/data/item_dummy.csv'
user_code_path = '/home/dhkim/Fragrance/data/user_code.pkl.csv'
X_path = '/home/dhkim/Fragrance/data/X.csv'
Y_path = '/home/dhkim/Fragrance/data/Y.csv'
field_dict_path = "/home/dhkim/Fragrance/data/field_dict.pkl"
field_index_path = "/home/dhkim/Fragrance/data/field_index.pkl"
checkpoint_filepath = "/home/dhkim/Fragrance/model/cp.ckpt"
classification_checkpoint_filepath = "/home/dhkim/Fragrance/model/classification/cp.ckpt"
regression_checkpoint_filepath = "/home/dhkim/Fragrance/model/regression/cp.ckpt"
field_dict_user_path = "/home/dhkim/Fragrance/data/field_dict_user.pkl"
field_dict_item_path = "/home/dhkim/Fragrance/data/field_dict_item.pkl"
user_code_path = '/home/dhkim/Fragrance/data/user_code.pkl'


types_list = ['Citrus','Leathery','Aquatic','Earthy','Oriental','Animal','Green',
'Spicy','Fresh','Chypre','Floral','Powdery','Synthetic','Smoky',
'Woody','Gourmand','Creamy','Resinous','Fougere', 'Sweet','Fruity']

types_dict ={'Citrus': 0,'Leathery': 1, 'Aquatic': 2, 'Earthy': 3,
'Oriental': 4,'Animal': 5,'Green': 6,
'Spicy': 7,'Fresh': 8,'Chypre': 9,
'Floral': 10,'Powdery': 11,'Synthetic': 12,
'Smoky': 13,'Woody': 14,'Gourmand': 15,
'Creamy': 16,'Resinous': 17,'Fougere': 18, 
'Sweet': 19,'Fruity': 20}

ALL_FIELDS = [ 'rating', 'Scent','Longevity', 'Sillage', 'Value for money', 

              'Spring', 'Summer', 'Fall','Winter', 

              'Old', 'Young', 'Men', 'Women', 

              'Leisure', 'Daily','Night Out', 'Business', 'Sport', 'Evening',

              'Earthy', 'Spicy', 'Powdery', 'Fruity','Resinous', 'Leathery', 'Sweet',
              'Oriental', 'Smoky', 'Synthetic', 'Chypre', 'Fougere', 'Animal','Gourmand',
              'Creamy', 'Aquatic', 'Citrus', 'Woody', 'Floral', 'Fresh','Green',

              'gender','year','brand','perfumer',
              'top_notes','base_notes','heart_notes',

              'count','user_gender', 'nation', 'user_id']

CONT_FIELDS = ['user_rating', 'rating']
CONT_FIELDS.extend(types_list)
CONT_FIELDS.extend(['Spring','Summer','Fall','Winter'])
CONT_FIELDS.extend(['Leisure','Daily','Night Out','Business','Sport','Evening'])
CONT_FIELDS.extend(['Old','Young','Men','Women'])
CONT_FIELDS.extend(['Scent','Longevity','Value for money','Sillage'])

CAT_FIELDS = list(set(ALL_FIELDS).difference(CONT_FIELDS))

test_size = 0.2
epochs = 200
embedding_size = 7
lr = 0.001
batch_size = 128
seed = 5
