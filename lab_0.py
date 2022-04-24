import numpy as np
import pandas as pd


def q1():
    matrix = np.zeros((7, 10))
    # print(matrix)
    matrix[::2, 1::2] = 1  # [start:stop:step]
    print(matrix)
    matrix[1::2, ::2] = 1
    print("\n")
    print(matrix)


def q2():
    diameters = np.array([1, 3, 5, 2, 4])
    lengths = np.array([10, 20, 3, 10, 5])
    radius_square = np.power(diameters * 0.5, 2)
    print(radius_square)
    volumes = radius_square * lengths * np.pi
    print(volumes, volumes.shape)


def create_cartesian_product(vec1, vec2):
    # v1 = np.array([1, 2])
    # v2 = np.array([3, 4, 5])
    # v1_repeat = np.repeat(v1, len(v2))
    # print("v1_rep: ", v1_repeat)
    # v2_tile = np.tile(v2, len(v1))
    # print("v2_tile: ", v2_tile)
    # concat = np.concatenate((np.array([v1_repeat]), np.array([v2_tile])))
    # print(concat)
    # print(np.transpose(concat))

    vec1_repeat = np.array([np.repeat(vec1, len(vec2))])
    vec2_tile = np.array([np.tile(vec2, len(vec1))])
    concat = np.concatenate((vec1_repeat, vec2_tile))
    return np.transpose(concat)
    # return np.transpose(np.array([np.repeat(vec1, len(vec2)), np.tile(vec2, len(vec1))]))


def find_closest(a, n):
    a = np.array(a)
    new_arr = np.abs(a - n)
    return a[(np.argsort(new_arr))[0]]


def check_sudoku(grid):
    grid = np.array(grid)
    valid_row = np.arange(1, 10)
    if not (np.all(np.sort(grid) == valid_row)):
        return False
    if not (np.all(np.sort(np.transpose(grid)) == valid_row)):
        return False


def check_dependencies(matrix_):
    return np.linalg.matrix_rank(matrix_) != len(matrix_)

def check_dependencies2(matrix_):
    def rows_are_dependent(indices):
        if indices[0] == indices[1]:
            return False
        # print("indices: ", indices, "\n")
        # print(matrix_[indices[0],])
        # print(matrix_[indices[1],])
        # print(matrix_[indices[0],] / matrix_[indices[1],])
        # print(np.unique(matrix_[indices[0],] / matrix_[indices[1],]).shape)
        return np.unique(matrix_[indices[0],] / matrix_[indices[1],]).shape[0] == 1

    print(np.apply_along_axis(rows_are_dependent, 1,
                              create_cartesian_product(np.arange(matrix_.shape[0]), np.arange(matrix_.shape[0]))))
    return np.any(np.apply_along_axis(rows_are_dependent, 1,
                                      create_cartesian_product(np.arange(matrix_.shape[0]), np.arange(matrix_.shape[0]))))


def have_a_maxima(array):
    argmax_arr = np.argmax(array[1:-1]) + 1
    print("argmax_arr: ", argmax_arr)
    print(array[:argmax_arr - 1])
    print(array[1:argmax_arr])
    print(array[:argmax_arr - 1] - array[1:argmax_arr])
    before_max_neg = np.all(array[:argmax_arr - 1] - array[1:argmax_arr] < 0)
    after_max_neg = np.all(array[argmax_arr:-1] - array[argmax_arr + 1:] > 0)
    return before_max_neg and after_max_neg


def have_an_extrema(array):
    # Check if it is a monotonic series
    # print(array[:-1])
    # print(array[1:])
    if np.unique(array[:-1] - array[1:]).shape[0] == 1: return True

    # If there is a minimum, we can look for a maximum in the negated array
    return have_a_maxima(array) or have_a_maxima(-array)

"pandas"

def create_flight_df(cities_poss, nrows = 20):
    flights = pd.DataFrame([], columns=['Departure', 'Destination', 'Price'])
    i = 0
    while i < nrows:
        dep_dest = np.random.choice(cities_poss, 2, False)
        if not ((flights["Departure"] == dep_dest[0]) & (flights["Destination"] == dep_dest[1])).any():
            flights = flights.append({"Departure": dep_dest[0], "Destination": dep_dest[1], "Price": np.random.randint(100, 401)}, ignore_index=True)
            i += 1

    df = pd.merge(flights, flights, left_on=["Destination"], right_on=["Departure"], how="inner")
    df = df[df.Departure_x != df.Destination_y]
    df["Total_Price"] = df["Price_x"] + df["Price_y"]

    single_connection = df[["Departure_x", "Destination_y", "Total_Price"]]
    single_connection_rename = single_connection.rename(columns={"Departure_x" : "Departure", "Destination_y" : "Destination", "Total_Price" : "Price"})
    max_single_connection = pd.concat([flights, single_connection_rename])

    min_by_group = single_connection.groupby(["Departure_x", "Destination_y"], as_index=False)["Total_Price"].min()

    dest_prices = min_by_group.groupby("Destination_y")["Total_Price"].mean()
    print(dest_prices)
    print(dest_prices.idxmax())


if __name__ == '__main__':
    # q1()
    # q2()
    # print(create_cartesian_product([1, 2, 3], [4, 5, 6, 7]))
    # print(find_closest([1, 24, 12, 13, 14], 10))

    # matrix_ = np.array([[1, 2],
    #                     [2, 4]])
    # print(check_dependencies2(matrix_))
    # print(check_dependencies(matrix_))

    # arr = np.array([1, 3, 6, 7, 4, 2, 3])
    # arr2 = np.array([1, 2, 3, 4, 7])
    # print(have_an_extrema(arr))
    # print(have_an_extrema(arr2))

    cities = ["Beijing", "Moscow", "New-York", "Tokyo", "Paris", "Cairo", "Santiago", "Lima", "Kinshasa", "Singapore",
              "New-Delhi", "London", "Ankara", "Nairobi", "Ottawa", "Seoul", "Tehran", "Guatemala", "Caracas", "Vienna"]
    print(create_flight_df(cities))


