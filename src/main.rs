#![allow(dead_code)]

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn variance(x: f64) -> f64 {
    x * (1. - x)
}

fn transponse<S, T, D>(v: S) -> Option<Vec<Vec<D>>>
where
    D: Copy + Default,
    T: AsRef<[D]>,
    S: AsRef<[T]>,
{
    let v = v.as_ref();
    v.first().and_then(|fir| {
        let mut result = vec![vec![Default::default(); v.len()]; fir.as_ref().len()];

        result.iter_mut().enumerate().for_each(|(row, val)| {
            val.iter_mut().enumerate().for_each(|(column, val2)| {
                *val2 = v.as_ref()[column].as_ref()[row];
            })
        });

        Some(result)
    })
}

fn test() {
    // We expect 0, 1, 1, 0, 0
    let input = vec![
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 1],
    ];

    // Got this weights from the train function
    let weights = [
        16.52178953510465,
        -8.064882571154394,
        -1.156219531292027,
        2.0143330904489054,
        -8.13440743275802,
        -1.201104782213791,
    ];

    let output = input
        .iter()
        .map(|&val| {
            let product = val
                .iter()
                .enumerate()
                .map(|(i, &val)| val as f64 * weights[i])
                .sum::<f64>();
            sigmoid(product).round()
        })
        .collect::<Vec<_>>();

    // prints `Results: [0.0, 1.0, 1.0, 0.0, 0.0]`
    println!("Results: {output:?}")
}

fn train() {
    let train_input = vec![
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 1],
    ];
    let t_train_in = transponse(&train_input).unwrap();

    let train_out = vec![0, 0, 1, 1, 0, 0, 1, 1, 0];
    let mut error = vec![0.; train_out.len()];
    let mut adjust = vec![0.; train_out.len()];

    let mut weights = [0.; 6];
    let mut final_out = vec![];
    weights
        .iter_mut()
        .for_each(|val| *val = 2. * rand::random::<f64>() - 1.);
    // println!("Weight: {weights:?}");

    for i in 0..=1_000_000 {
        let output = train_input
            .iter()
            .map(|&val| {
                let product = val
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| val as f64 * weights[i])
                    .sum::<f64>();
                sigmoid(product)
            })
            .collect::<Vec<_>>();

        if i == 1_000_000 {
            final_out = output.clone();
        }

        // println!("Out: {output:?}");
        error
            .iter_mut()
            .enumerate()
            .for_each(|(i, val)| *val = train_out[i] as f64 - output[i]);
        // println!("Error: {error:?}");

        adjust
            .iter_mut()
            .enumerate()
            .for_each(|(i, val)| *val = error[i] * variance(output[i] as f64));
        // println!("Adjust: {adjust:?}");

        let synaptic_weights = t_train_in
            .iter()
            .map(|val| {
                val.iter()
                    .enumerate()
                    .map(|(i, &val)| val as f64 * adjust[i])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        // println!("SynWeight: {synaptic_weights:?}");

        weights.iter_mut().enumerate().for_each(|(i, val)| {
            *val += synaptic_weights[i];
        });
        // println!("FWeight: {weights:?}");
    }
    println!("Weight: {weights:?}");
    println!("Out: {final_out:?}");
}

fn main() {
    // train();
    test();
}
