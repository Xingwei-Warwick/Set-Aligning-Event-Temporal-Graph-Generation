from nyt_parser import NYTArticle
from os import listdir
from os.path import isfile
import json
from argparse import ArgumentParser


FILTER_LIST = ["bomb", "terrorism", "murder", "riots", "hijacking", "assassination", "kidnapping", "arson", "vandalism", "hate crime", "serial murder", "manslaughter", "extortion"]
FILTER_SET = set(FILTER_LIST)


def load_NYT(select_from_ids_file=None):
    if select_from_ids_file is not None:
        with open(select_from_ids_file, 'r') as f:
            selected_ids = json.loads(f.read())

    count_by_year = []
    data_dict = {}
    sentence_count = 0
    total_selected = 0
    descriptors_count = {}
    descriptors_doc = {}
    for i in range(1987, 2008):
        count = 0

        for month_name in listdir(f"NYT_annotated/data/{i}"):
            if isfile(f"NYT_annotated/data/{i}/{month_name}"):
                continue
            for date_name in listdir(f"NYT_annotated/data/{i}/{month_name}"):
                if isfile(f"NYT_annotated/data/{i}/{month_name}/{date_name}"):
                    continue
                for file_name in listdir(f"NYT_annotated/data/{i}/{month_name}/{date_name}"):
                    if file_name.split('.')[-1] == 'xml':
                        f_id = file_name.split('.')[0]

                        if select_from_ids_file is not None and f"{i}_{month_name}_{date_name}_{f_id}" not in selected_ids:
                            continue

                        with open(f"NYT_annotated/data/{i}/{month_name}/{date_name}/{file_name}", 'r', encoding='utf8') as f:
                            this_doc = NYTArticle.from_file(f)
                        this_doc = this_doc.as_dict()
                        this_descriptors = []

                        for d in this_doc["descriptors"]:
                            this_descriptors.append(d)
                        this_descriptors = set(this_descriptors)

                        for des in this_descriptors:
                            if descriptors_count.get(des) is None:
                                descriptors_count[des] = 0
                            descriptors_count[des] += 1
                            if descriptors_doc.get(des) is None:
                                descriptors_doc[des] = []
                            if len(descriptors_doc[des]) > 1000:
                                continue
                            descriptors_doc[des].append('\n'.join(this_doc["paragraphs"]))
                        

                        data_dict[f"{i}_{month_name}_{date_name}_{f_id}"] = this_doc
                        count += 1
                        total_selected += 1
                        sentence_count += len(this_doc["paragraphs"])
        count_by_year.append(count)

    print(count_by_year)
    print(sentence_count)
    print(total_selected)    
    print(f"Load {len(data_dict)} files!")
    return data_dict

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--select_from_ids_file", type=str, required=True, help="File containing list of ids to select from NYT data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output path to save the selected data")
    args = parser.parse_args()

    data_dict =  load_NYT(args.select_from_ids_file)

    # with open(args.output_path, 'w') as f:
    #     f.write(json.dumps(data_dict))

    for key in data_dict:
        doc_text = '\n'.join(data_dict[key]["paragraphs"])
        with open(f"{args.output_dir}/{key}.txt", 'w') as f:
            f.write(doc_text)