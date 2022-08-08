{
  "line_of_i": "what thing is <thing> given <other_thing> where <third>"
  "vars": {
    "thing": ["apple", "orange"],
    "other_thing": ["abcd"],
    "third": ["a","b","c"]
  }
}

python

def validate_json(js):
  # check if loi is in json, if not give clear error message
  # check vars exitst
  # check all var reference in loi exist in json
  # make sure all elements in vars are lists

def parse_loi(splits, js):
"""
this function takes in the line of inquiry and generates a dict with each var's name and index in the string split
"""
  element_indexes = {}
  for i in js["vars"]:
    element_indexes[i] = splits.index(i) # get index of vars element in loi split list
    # ie: element_indexes = {"thing": 2, "other_thing": 4, "third: 6"}
  return element_indexes

if __main__:
  import json
  with open("path_to_j.json", "r") as file:
    my_json = file.read()
    input_dict = json.dumps(my_json)

  validate_json(js) # make sure our json is valid before running any code
  line_of_i = input_dict["line_of_i"]
  vars = input_dict["vars"]
  splits = line_of_i.split(" ")

  # get our indexes in the loi splits
  element_indexes = parse_loi(splits, my_json)

  specific_questions = []
  for w in splits:
    if vars.get(w) is not None: # check if current word is in vars
