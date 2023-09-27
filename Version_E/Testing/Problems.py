

def make_problem(which, size: int):
    if size == "small":
        amount_of_bits = 5
    elif size == "medium":
        amount_of_bits = 25
    elif size == "large":
        amount_of_bits = 100
    else:
        amount_of_bits = 20


    if which == "binval":
        return {"which": which,
                "size": amount_of_bits,
                "base": 2}
    elif which == "onemax":
        return {"which": which,
                "size": amount_of_bits}
    elif which == "trapk":
        return {"which": which,
                "k": 5,
                "amount_of_groups:": amount_of_bits // 5}
    elif which == "checkerboard":
        if size == "small":
            rows = 4
        elif size == "medium":
            rows = 8
        else:
            rows = 16
        return {"which": which,
                "rows": rows,
                "cols": rows}
    elif which == "artificial":
        return {"which": which,
                "size": amount_of_bits,
                "size_of_partials": 5,
                "amount_of_features": 5,
                "allow_overlaps": True}
    elif which == "knapsack":
        return {"which": which,
                "expected_price": 50,
                "expected_weight": 1000,
                "expected_volume": 15}
    elif which == "graph":
        return {"which": which,
                "amount_of_colours": 4,
                "amount_of_nodes": amount_of_bits,
                "chance_of_connection": 0.3}
    else:
        raise Exception("Problem string was not recognised")



