
# removing unknown values from list
def remove_unknown(lst):
    lst = list(set(lst))
    if 0 in lst:
        lst.remove(0) # code for unknown diag
    if 500 in lst:
        lst.remove(500) # code for unknown sick
    if 768 in lst:
        lst.remove(768) # code for unknown pres
    if len(lst)==0:
        return [0] # if list is empty (which is highly unlikely), we return [0] instead of an empty list to prevent future errors
    return lst

# function for getting an index for diag, sick, pres
def to_dict_idx(string, data_type, D):
    # string: input string, data_type: ['diag','sick','pres']
    # D: corresponding dictionary, different dictionary entered
    if data_type=='diag':
        try:
            out = D[string]
            return out
        except KeyError:
            return 0
    elif data_type=='sick':
        try:
            out = get_classified_sickness(string, D[0], D[1])
            return out + 500
        except KeyError:
            return 500
    elif data_type=='pres':
        # print("start:" ,string)
        string = string[:4]
        try:
            out = D[string]
            # print("Out: ",out)
            return out + 768
        except KeyError:
            # print("Error")
            return 768

# function for obtaining sickness type
def get_classified_sickness(sample, alphabet_dict1, alphabet_dict2):
    # if input is not in the form of such as 'A54'
    if len(sample)<3:
        return 0
    c, num = sample[0], int(sample[1:]) # c: character, num: 2-digit int
    for i, rng in enumerate(alphabet_dict1[c]):
        if '-' in rng: # if rng is a range such as 'A40-A49'
            lower,upper = rng.split('-')
            lower = int(lower[1:]) # remove alphabet
            upper = int(upper[1:])
            if (lower<=num) & (upper>=num): # if number is in this range
                answer = alphabet_dict2[c][i]
                return answer
        else: # if rng is a number
            if num==int(rng[1:]):
                answer = alphabet_dict2[c][i]
                return answer
    return 0 # not in here
