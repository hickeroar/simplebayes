from concurrent.futures import ThreadPoolExecutor

from simplebayes import SimpleBayes


def test_parallel_train_and_score_completes():
    classifier = SimpleBayes()

    def train_and_score(index: int) -> None:
        classifier.train("tech", f"python fastapi service sample {index}")
        _ = classifier.score("python service")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(train_and_score, range(50)))

    result = classifier.classify_result("python service")
    assert result.category == "tech"
    assert result.score > 0
    summaries = classifier.get_summaries()
    assert summaries["tech"].token_tally == 250
    assert classifier.tally("tech") == 250


def test_parallel_classify_during_mutation():
    classifier = SimpleBayes()
    classifier.train("alpha", "one two three")
    classifier.train("beta", "four five six")

    def mutate() -> None:
        for _ in range(50):
            classifier.train("alpha", "one two three")
            classifier.untrain("alpha", "one")

    def classify() -> None:
        for _ in range(50):
            _ = classifier.classify("one five")
            _ = classifier.get_summaries()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(mutate), pool.submit(classify), pool.submit(classify)]
        for future in futures:
            future.result()

    assert classifier.tally("alpha") == 103
    assert classifier.tally("beta") == 3
    summaries = classifier.get_summaries()
    assert summaries["alpha"].token_tally == 103
    assert summaries["beta"].token_tally == 3
    assert abs((summaries["alpha"].prob_in_cat + summaries["alpha"].prob_not_in_cat) - 1.0) < 1e-12
