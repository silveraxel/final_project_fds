import matplotlib.pyplot as plt

def genre_distribution():

    generi = [
        "Drama", "Comedy", "Thriller", "Action", "Romance",
        "Adventure", "Crime", "Sci-Fi", "Horror", "Fantasy",
        "Children", "Animation", "Mystery", "Documentary", "War",
        "Musical", "Western", "IMAX", "Film-Noir", "(no genres listed)"
    ]

    valori = [
        4361, 3756, 1894, 1828, 1596,
        1263, 1199, 980, 978, 779,
        664, 611, 573, 440, 382,
        334, 167, 158, 87, 34
    ]

    plt.figure(figsize=(14, 6))
    plt.bar(generi, valori)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Numero")
    plt.title("Genre Distribution")

    plt.tight_layout()

    plt.savefig("analisi/genre_distribuition.png", dpi=300, bbox_inches="tight")

    plt.close()


def decade_distribuition():

    generi = [
        "1900", "1910", "1920", "1930", "1940",
        "1950", "1960", "1970", "1980", "1990",
        "2000", "2010"
    ]

    valori = [
        3, 7, 37, 136, 197, 279, 401, 500, 1177, 2212, 2849, 1931
    ]

    plt.figure(figsize=(14, 6))
    plt.bar(generi, valori)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Numero")
    plt.title("Decade Distribution")

    plt.tight_layout()

    plt.savefig("analisi/decade_distribuition.png", dpi=300, bbox_inches="tight")

    plt.close()


def top_tag():

    generi = [
        "In Netflix queue", "atmospheric", "thought-provoking", "superhero", "surreal",
        "funny", "Disney", "religion", "quirky", "sci-fi"
    ]

    valori = [
       131, 36, 24, 24, 23, 23, 23, 22, 21, 21
    ]

    plt.figure(figsize=(14, 6))
    plt.bar(generi, valori)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Numero")
    plt.title("Top 10 Frequency Tag")

    plt.tight_layout()

    plt.savefig("analisi/top_frequency_tag.png", dpi=300, bbox_inches="tight")

    plt.close()



if __name__ == "__main__":
    genre_distribution()
    decade_distribuition()
    top_tag()