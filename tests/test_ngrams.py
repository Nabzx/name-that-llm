from nthl.ngrams import four_views, function_words, word_shape


def test_word_shape_keeps_case_digits_and_punctuation():
    assert word_shape("Hi, World 42!") == "Xx , Xxxxx dd !"


def test_word_shape_caps_long_runs_at_four():
    assert word_shape("elephant") == "xxxx"
    assert word_shape("World") == "Xxxxx"
    assert word_shape("NASA") == "XXXX"


def test_word_shape_reads_accents_as_letters():
    assert word_shape("Gödel") == "Xxxxx"


def test_function_words_hide_the_topic():
    assert function_words("The cat sat, quietly.") == "The CONTENT CONTENT , CONTENT ."


def test_function_words_keep_curly_quotes_and_dashes():
    assert function_words("it’s here—now") == "it ’ s here — now"


def test_four_views_stack_into_one_matrix():
    docs = ["The cat sat — quietly.", "A dog ran, fast!", "It’s 42 here."]
    views = four_views(min_df=1)
    x = views.fit_transform(docs)
    assert [name for name, _ in views.transformer_list] == ["char", "word", "shape", "function"]
    assert x.shape[0] == 3
    assert x.shape[1] == sum(len(v.vocabulary_) for _, v in views.transformer_list)
