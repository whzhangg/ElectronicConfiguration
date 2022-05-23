# defines a similarity between atoms based on the periodic table
from mldos.parameters import defined_symbol_tuple_dict

reversed_dict = { t:s for s,t in defined_symbol_tuple_dict.items() }
similarity_dict_7 = {}
for ele, position in defined_symbol_tuple_dict.items():
    i,j = position

    first = []
    if (i + 1, j) in reversed_dict:
        first.append(reversed_dict[(i + 1, j)])
    if (i - 1, j) in reversed_dict:
        first.append(reversed_dict[(i - 1, j)])

    second = []
    if (i + 2, j) in reversed_dict:
        second.append(reversed_dict[(i + 2, j)])
    if (i - 2, j) in reversed_dict:
        second.append(reversed_dict[(i - 2, j)])
    if (i, j + 1) in reversed_dict:
        second.append(reversed_dict[(i, j + 1)])
    if (i, j - 1) in reversed_dict:
        second.append(reversed_dict[(i, j - 1)])
    
    similarity_dict_7.setdefault(ele, {"1st": first, "2nd": second})

similarity_dict_5 = {}
for ele, position in defined_symbol_tuple_dict.items():
    i,j = position

    first = []
    if (i + 1, j) in reversed_dict:
        first.append(reversed_dict[(i + 1, j)])
    if (i - 1, j) in reversed_dict:
        first.append(reversed_dict[(i - 1, j)])
    if (i, j + 1) in reversed_dict:
        first.append(reversed_dict[(i, j + 1)])
    if (i, j - 1) in reversed_dict:
        first.append(reversed_dict[(i, j - 1)])
    
    similarity_dict_5.setdefault(ele, first)

# print(similarity_dict["Sn"])
# >>> {'1st': ['Pb', 'Ge'], '2nd': ['Si', 'Sb', 'In']}

