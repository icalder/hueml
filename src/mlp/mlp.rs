use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

use crate::mlp::fns;

use super::config::{MLPConfig, TrainingState};

// Neural Network From Scratch: No Pytorch & Tensorflow; just pure math | 30 min theory + 30 min coding
// https://youtu.be/A83BbHFoKb8?si=9zzxB-IFyPYYwV9c

// Neural Networks From Scratch in Rust
// https://www.youtube.com/watch?v=DKbz9pNXVdE&t=23s

#[derive(Serialize, Deserialize)]
pub struct MLP {
    config: MLPConfig,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    // Used to store hidden layer [A] values
    #[serde(skip)]
    a: Vec<Array2<f64>>,
}

impl MLP {
    pub fn new(config: MLPConfig) -> Self {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..config.layers.len() - 1 {
            weights.push(Array2::random(
                (config.layers[i + 1], config.layers[i]),
                Uniform::new(-0.5, 0.5),
            ));
            biases.push(Array2::random(
                (config.layers[i + 1], 1),
                Uniform::new(-0.5, 0.5),
            ));
        }

        Self {
            config,
            weights,
            biases,
            a: vec![],
        }
    }

    fn outputfn(&self) -> fn(&Array2<f64>) -> Array2<f64> {
        if let Some(l) = self.config.layers.last() {
            // Networks with a single output neuron use sigmoid
            if *l == 1 {
                return fns::sigmoid;
            }
        }
        fns::softmax
    }

    pub fn load(
        path: &str,
        training_state_updated: Option<fn(TrainingState)>,
    ) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let mut mlp: MLP = serde_json::from_reader(file)?;
        mlp.config.training_state_updated = training_state_updated;
        Ok(mlp)
    }

    // 3 layers e.g. 2[x],3[h],1[y]
    // w1 from x to the hidden layer
    // w2 from hidden layer to output
    // z1 = w1 * x + b1
    // a1 = f(z1)
    // z2 = w2 * a1 + b2
    // y = fout(z2)

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Array2<f64> {
        assert!(
            self.config.layers[0] == inputs.len(),
            "Invalid number of inputs"
        );

        // self.a[0] is X
        let mut an = Array::from_shape_vec((self.config.layers[0], 1), inputs).unwrap();
        self.a = vec![an.clone()];
        for i in 0..self.config.layers.len() - 1 {
            let zn = &self.weights[i].dot(&an) + &self.biases[i];
            an = zn.mapv(|v| (self.config.activation.function)(&v));
            if i == self.config.layers.len() - 1 {
                an = self.outputfn()(&an);
            }
            self.a.push(an.clone());
        }
        an
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        let mut mse: f64 = 0.0;
        for i in 1..=epochs {
            for j in 0..inputs.len() {
                self.feed_forward(inputs[j].clone());
                mse = self.back_propagate(
                    //self.a[self.config.layers.len() - 1].clone(),
                    Array::from_shape_vec(
                        (self.config.layers[self.config.layers.len() - 1], 1),
                        targets[j].clone(),
                    )
                    .unwrap(),
                );
            }
            if epochs < 100 || i % (epochs / 100) == 0 {
                if let Some(callback) = &self.config.training_state_updated {
                    callback(TrainingState {
                        total_epochs: epochs,
                        epoch: i,
                        mse,
                    });
                }
            }
        }
    }

    // fn back_propagatex(&mut self, outputs: Array2<f64>, targets: Array2<f64>) -> f64 {
    //     let mut errors = targets - &outputs;
    //     let mut gradients = outputs.map(self.config.activation.derivative);
    //     for i in (0..self.config.layers.len() - 1).rev() {
    //         gradients = gradients * (&errors).mapv(|v| v * self.config.learning_rate);
    //         self.weights[i] += &gradients.dot(&self.a[i].t());
    //         self.biases[i] += &gradients;
    //         errors = self.weights[i].t().dot(&errors);
    //         gradients = self.a[i].map(self.config.activation.derivative);
    //     }
    //     // return mean-square error
    //     errors = errors.mapv(|x| x * x);
    //     errors.sum() / errors.len() as f64
    // }

    fn zl(&self, l: usize) -> Array2<f64> {
        // NB: a[l] works here instead of a[l-1] because a[0] = x
        self.weights[l].dot(&self.a[l]) + &self.biases[l]
    }

    // x(a0) -> a1=sigma(z1) -> a2=sigma(z2)

    // http://neuralnetworksanddeeplearning.com/chap2.html
    // https://youtu.be/tIeHLnjs5U8?si=LYWn7ZYKv6FrOgcg
    // zl = wl.(al−1) + bl
    fn back_propagate(&mut self, y: Array2<f64>) -> f64 {
        // sigma-prime: the derivitive of the activation function
        let sigmap = self.config.activation.derivative;
        // start with the last layer - e.g. l=2 for 3 layers (1 hidden layer) [0,1,2]
        // NB: weight and bias layers are [0,1]
        let l = self.config.layers.len() - 1;
        let mut deltal = (&self.a[l] - &y) * self.zl(l - 1).mapv(|v| sigmap(&v));
        let mut dw = deltal.dot(&self.a[l - 1].t());
        let mut db = &deltal;
        // see here for other optimisation algorithms: https://towardsdatascience.com/neural-network-optimizers-from-scratch-in-python-af76ee087aab/
        self.weights[l - 1] -= &dw.mapv(|v| v * self.config.learning_rate);
        self.biases[l - 1] -= &db.mapv(|v| v * self.config.learning_rate);

        for l in (0..self.config.layers.len() - 2).rev() {
            deltal = self.weights[l + 1].t().dot(&deltal) * self.zl(l).mapv(|v| sigmap(&v));
            // a[l] works here instead of a[l-1] because a[0] = x
            dw = deltal.dot(&self.a[l].t());
            db = &deltal;
            self.weights[l] -= &dw.mapv(|v| v * self.config.learning_rate);
            self.biases[l] -= &db.mapv(|v| v * self.config.learning_rate);
        }
        // return mean-square error
        deltal = deltal.mapv(|x| x * x);
        deltal.sum() / deltal.len() as f64
    }

    pub fn dump(&self, path: &str) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(path)?;
        serde_json::to_writer(&mut file, self)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::mlp::{self, config::MLPConfig, mlp::MLP};

    #[test]
    fn test_for_range_loop() {
        let i = 1;
        for i in (0..i).rev() {
            println!("{}", i);
        }
    }

    #[test]
    fn test_mlp_serialize_to_file() {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        let mut mlp = MLP::new(MLPConfig {
            layers: vec![2, 3, 1],
            activation: mlp::fns::LOGISTIC,
            learning_rate: 0.5,
            training_state_updated: None,
        });

        mlp.train(inputs, targets, 10000);
        mlp.dump("data/mlp.json").unwrap();

        let mut mlp = MLP::load("data/mlp.json", None).unwrap();
        println!("{:?}", mlp.feed_forward(vec![0.0, 0.0]));
        println!("{:?}", mlp.feed_forward(vec![0.0, 1.0]));
        println!("{:?}", mlp.feed_forward(vec![1.0, 0.0]));
        println!("{:?}", mlp.feed_forward(vec![1.0, 1.0]));
    }

    #[test]
    fn test_mlp() {
        // Train for XOR
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        let mut mlp = MLP::new(MLPConfig {
            layers: vec![2, 3, 1],
            activation: mlp::fns::LOGISTIC,
            learning_rate: 0.1,
            training_state_updated: None,
        });

        mlp.train(inputs, targets, 100000);

        println!("{:?}", mlp.feed_forward(vec![0.0, 0.0]));
        println!("{:?}", mlp.feed_forward(vec![0.0, 1.0]));
        println!("{:?}", mlp.feed_forward(vec![1.0, 0.0]));
        println!("{:?}", mlp.feed_forward(vec![1.0, 1.0]));

        /*
        [[0.010837591978139343]], shape=[1, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2
        [[0.9896457406477849]], shape=[1, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2
        [[0.989648726249577]], shape=[1, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2
        [[0.010077380921864295]], shape=[1, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2
                                 */
    }
}
