# go through all summaries
# check if they exist in extracted folder file list
# if not, dont copy to a new folder
import os

extracted_filenames = os.listdir('extracted')
for original_filename in os.listdir('summaries'):
    if original_filename not in extracted_filenames:
        os.remove('summaries/' + original_filename)
        print('removed ' + original_filename)
print('done')
