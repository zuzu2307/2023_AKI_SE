import Levenshtein


def search(source, target, s_lenght, t_lenght):
    # for recursive to start point
    if s_lenght == 0:
        return t_lenght

    if t_lenght == 0:
        return s_lenght

    # if it is the same alphabet return diagonal
    if source[s_lenght - 1] == target[t_lenght - 1]:
        return search(source, target, s_lenght - 1, t_lenght - 1)

    # return minimum of distance between 3 type of edit DELETE, ADD, CHANGE and + 1 for action
    return 1 + min(search(source, target, s_lenght - 1, t_lenght), min(search(source, target, s_lenght, t_lenght - 1), search(source, target, s_lenght - 1, t_lenght - 1)))


source = 'I am father'
target = 'I am mother'

# Compare between library and created function
lib_distance = Levenshtein.distance(source, target)
distance = search(source, target, len(source), len(target))

print('Library Edit times is ' + str(lib_distance))
print('My Edit times is ' + str(distance))
