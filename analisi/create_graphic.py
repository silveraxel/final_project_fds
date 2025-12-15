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
    #plt.ylabel("Numero")
    plt.title("Genre Distribution")

    plt.tight_layout()

    plt.savefig("analisi/genre_distribution.png", dpi=300, bbox_inches="tight")

    plt.close()


def decade_distribution():

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
    #plt.ylabel("Numero")
    plt.title("Decade Distribution")

    plt.tight_layout()

    plt.savefig("analisi/decade_distribution.png", dpi=300, bbox_inches="tight")

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
    #plt.ylabel("Numero")
    plt.title("Top 10 Frequency Tag")

    plt.tight_layout()

    plt.savefig("analisi/top_frequency_tag.png", dpi=300, bbox_inches="tight")

    plt.close()



def rating_distribution():

    generi = [
        "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"
    ]

    valori = [
       1370, 2811, 1791, 7551, 5550, 20047, 13136, 26818, 8551, 13211
    ]

    plt.figure(figsize=(14, 6))
    plt.bar(generi, valori)

    plt.xticks(rotation=45, ha="right")
    #plt.ylabel("Numero")
    plt.title("Rating Distribution")

    plt.tight_layout()

    plt.savefig("analisi/rating_distribution.png", dpi=300, bbox_inches="tight")

    plt.close()

def pie_user_film_rating():
    names = ["User", "Film", "Rating"]
    values = [610, 9724, 100836]

    labels = [
        f"User ({values[0]})",
        f"Film ({values[1]})",
        f"Rating ({values[2]})"
    ]

    colors = ["#FFBB00", "#37FF00", "#0044FF"]

    plt.figure(figsize=(7, 7))
    plt.pie(
        values,
        colors=colors,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.axis("equal")

    plt.tight_layout()
    plt.savefig("analisi/pie_node.png", dpi=300, bbox_inches="tight")
    plt.close()




if __name__ == "__main__":
    genre_distribution()
    decade_distribution()
    top_tag()
    rating_distribution()
    pie_user_film_rating()