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
    // Use to store hidden layer [A] values
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

        let mut an = Array::from_shape_vec((self.config.layers[0], 1), inputs).unwrap();
        self.a = vec![];
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

    // fn back_propagate(&mut self, outputs: Array2<f64>, targets: Array2<f64>) -> f64 {
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

    // 3 layers e.g. 2[x],3[h],1[y]
    // w1 from x to the hidden layer
    // w2 from hidden layer to output
    // z1 = w1 * x + b1
    // a1 = f(z1)
    // z2 = w2 * a1 + b2
    // y = fout(z2)

    // https://drive.google.com/file/d/1QP64p8MGBfTHwucKCRFJxTRHfHdl9JLU/view
    fn back_propagate(&mut self, y: Array2<f64>) -> f64 {
        // i is used to index a[] and w[] - these have size nlayers - 1 i.e. 0 .. nlayers - 2
        let mut i = self.config.layers.len() - 2;

        let mut dz = &self.a[i] - &y;
        let mut dw = dz.dot(&self.a[i - 1].t());
        let mut db = dz.sum();
        self.weights[i] = &self.weights[i] - dw * self.config.learning_rate;
        self.biases[i] -= db * self.config.learning_rate;

        i -= 1;
        while i > 0 {
            // we need the derivative of the activation fn of the previous z layer
            let z = &self.weights[i].dot(&self.a[i]) + &self.biases[i];
            dz = self.weights[i + 1].t().dot(&dz)
                * z.mapv(|x| (self.config.activation.derivative)(&x));
            dw = dz.dot(&self.a[i].t());
            db = dz.sum();
            self.weights[i] = &self.weights[i] - dw * self.config.learning_rate;
            self.biases[i] -= db * self.config.learning_rate;
            i -= 1
        }

        dz = dz.mapv(|x| x * x);
        dz.sum() / dz.len() as f64
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
    fn test_mlp_serialize_to_file() {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]];

        let mut mlp = MLP::new(MLPConfig {
            layers: vec![2, 3, 1],
            activation: mlp::fns::LOGISTIC,
            learning_rate: 0.5,
            training_state_updated: None,
        });

        mlp.train(inputs, targets, 100000);
        mlp.dump("data/mlp.json").unwrap();

        let mut mlp = MLP::load("data/mlp.json", None).unwrap();
        println!("{:?}", mlp.feed_forward(vec![0.0, 0.0]));
        println!("{:?}", mlp.feed_forward(vec![0.0, 1.0]));
        println!("{:?}", mlp.feed_forward(vec![1.0, 0.0]));
        println!("{:?}", mlp.feed_forward(vec![1.0, 1.0]));
    }

    #[test]
    fn test_mlp() {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]];

        let mut mlp = MLP::new(MLPConfig {
            layers: vec![2, 3, 1],
            activation: mlp::fns::LOGISTIC,
            learning_rate: 0.5,
            training_state_updated: None,
        });

        mlp.train(inputs, targets, 100000);

        println!("{:?}", mlp.feed_forward(vec![0.0, 0.0]));
        println!("{:?}", mlp.feed_forward(vec![0.0, 1.0]));
        println!("{:?}", mlp.feed_forward(vec![1.0, 0.0]));
        println!("{:?}", mlp.feed_forward(vec![1.0, 1.0]));

        /*
        [[0.001975319636582013]], shape=[1, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2
        [[0.9983015657548819]], shape=[1, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2
        [[0.0019549341093425374]], shape=[1, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2
        [[0.9982879441701336]], shape=[1, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2
                 */
    }
}
