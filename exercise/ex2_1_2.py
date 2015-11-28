# exercise 2.1.2
from tmgsimple import TmgSimple
import tmgsimple
#help(tmgsimple)
fn='C:\\Users\\Bahram\\PycharmProjects\\Machine-Learning-and-Data-Mining\\02450Toolbox_Python\\Data\\textDocs.txt'
stopwords='C:\\Users\\Bahram\\PycharmProjects\\Machine-Learning-and-Data-Mining\\02450Toolbox_Python\\Data\\stopWords.txt'
tm = TmgSimple(filename=fn,stopwords_filename=stopwords,stem=True,min_term_length=5)
attributeNames = tm.get_words(sort=True)
x=tm.get_matrix(sort=True)

print attributeNames
print x


"""

# Generate text matrix with help of simple class TmgSimple


# Extract variables representing data
X = tm.get_matrix(sort=True)


# Display the result
print attributeNames
print X
"""