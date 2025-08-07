from opto.trainer.loader import DataLoader



def run_for_loop(dataloader):
    print('Running for-loop')
    for i, (inputs, infos) in enumerate(dataloader):

        print(f"Inputs: {inputs}, Infos: {infos}")

        if i == 0:
            assert inputs == [1, 2], f"First batch should contain inputs 1 and 2. Get: {inputs}"
            assert infos == ['a', 'b'], f"First batch should contain infos 'a' and 'b'. Get: {infos}"
        elif i == 1:
            assert inputs == [3, 4], f"Second batch should contain inputs 3 and 4. Get: {inputs}"
            assert infos == ['c', 'd'], f"Second batch should contain infos 'c' and 'd'. Get: {infos}"
        elif i == 2:
            assert inputs == [5], f"Third batch should contain input 5. Get: {inputs}"
            assert infos == ['e'], f"Third batch should contain info 'e'. Get: {infos}"

def run_next(dataloader):
    inputs, infos = next(dataloader)
    print('Running next()')
    print(f"Inputs: {inputs}, Infos: {infos}")

    assert inputs == [1, 2], f"First batch should contain inputs 1 and 2. Get: {inputs}"
    assert infos == ['a', 'b'], f"First batch should contain infos 'a' and 'b'. Get: {infos}"

    inputs, infos = next(dataloader)
    print(f"Inputs: {inputs}, Infos: {infos}")

    assert inputs == [3, 4], f"Second batch should contain inputs 3 and 4. Get: {inputs}"
    assert infos == ['c', 'd'], f"Second batch should contain infos 'c' and 'd'. Get: {infos}"

    inputs, infos = next(dataloader)
    print(f"Inputs: {inputs}, Infos: {infos}")

    assert inputs == [5], f"Third batch should contain input 5. Get: {inputs}"
    assert infos == ['e'], f"Third batch should contain info 'e'. Get: {infos}"

    try:
        next(dataloader)
    except StopIteration:
        print("No more data to iterate over, as expected.")

def run_sample(dataloader):

    print('Running sample()')
    inputs, infos = dataloader.sample()
    assert inputs == [1, 2], f"First sample should contain inputs 1 and 2. Get: {inputs}"
    assert infos == ['a', 'b'], f"First sample should contain infos 'a' and 'b'. Get: {infos}"
    inputs, infos = dataloader.sample()
    assert inputs == [3, 4], f"Second sample should contain inputs 3 and 4. Get: {inputs}"
    assert infos == ['c', 'd'], f"Second sample should contain infos 'c' and 'd'. Get: {infos}"
    inputs, infos = dataloader.sample()
    assert inputs == [5], f"Third sample should contain input 5. Get: {inputs}"
    assert infos == ['e'], f"Third sample should contain info 'e'. Get: {infos}"

    # At this point, the dataloader should be reset. No need to catch StopIteration when calling sample again

def test_dataloader():

    dataset = {
        'inputs': [1, 2, 3, 4, 5],
        'infos': ['a', 'b', 'c', 'd', 'e']
    }
    dataloader = DataLoader(dataset, batch_size=2, randomize=False)

    # Test for-loop usage
    run_for_loop(dataloader)
    run_for_loop(dataloader)  # make sure it can be iterated multiple times

    # Test next() usage
    run_next(dataloader)
    run_next(dataloader)  # make sure it can be called multiple times

    # Test sample() method
    run_sample(dataloader)
    run_sample(dataloader)  # make sure it can be called multiple times

    # Test for-loop usage
    run_for_loop(dataloader)
    run_for_loop(dataloader)  # make sure it can be iterated multiple times

    # Test next() usage
    run_next(dataloader)
    run_next(dataloader)  # make sure it can be called multiple times

    # Test sample() method
    run_sample(dataloader)
    run_sample(dataloader)  # make sure it can be called multiple times