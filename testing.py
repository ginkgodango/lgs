def FastModularExponentiation(b, k, m):
    k = 2 ** k
    if m == 1:
        c = 0
    else:
        c = 1
        b = b % m
        while k > 0:
            if k % 2 == 1:
                c = (c * b) % m
            k = k / 2
            b = (b * b) % m
    return c


assert FastModularExponentiation(
    3973588509869879969593491988475866750048006929901833594696821790176475421220219103437953024314400864,
    458,
    7205366042) == 2014068870


def FastModularExponentiation(b, e, m):
    c = 1
    while e > 1:
        if e & 1:
            c = (c * b) % m
        b = b ** 2 % m
        e >>= 1
    return (c * b) % m


assert FastModularExponentiation(
    8808956117332644023498092340849139128910822037356425708331689837907890426068066544248992840862152524,
    7490241720610468734920195456500065340699404657219686467800135578602755508447212245274650911833616801,
    1739635774) == 841325222
