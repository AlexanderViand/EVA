# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from eva import EvaProgram, Input, Output, evaluate, save, load
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse
import os
import time
import copy
import math
import numpy as np
import pandas as pd
from numpy import random

import known_type_pb2
import eva_pb2

####################
# BENCHMARKING     #
####################
times = {
    't_keygen': [],
    't_input_encryption': [],
    't_computation': [],
    't_decryption': []
}


def delta_ms(t0, t1):
    return round(1000 * abs(t0 - t1))


all_times = []
cur_times = []

# Generate Data
image_size = 32 * 32
layer1_units = 32
layer2_units = 16

# Fix seed so we can compare result in c++ more easily
random.seed(0)

# Input image
image = [0.5] * image_size

# Considering images padded to 32x32 for easier math
weights_1 = np.random.rand(layer1_units, image_size) - 0.5
bias_1 = np.random.rand(layer1_units) - 0.5

# Allowing 16 output classes, where the 6 extra ones are always zero/ignored
weights_2 = np.random.rand(layer2_units, layer1_units) - 0.5
bias_2 = np.random.rand(layer2_units) - 0.5


def diag(matrix, d):
    m, n = matrix.shape
    r = [0] * n
    for k in range(n):
        r[k] = matrix[k % m][(k + d) % n]
    return r


def mvp(ptxt_matrix, enc_vector):
    m, n = ptxt_matrix.shape

    log2_n_div_m = math.ceil(math.log(n // m, 2))
    t = 0
    for i in range(m):
        t += (enc_vector << i) * diag(ptxt_matrix, i)

    # TODO: if n/m isn't a power of two, we need to masking/padding here
    for i in range(log2_n_div_m):
        offset = n // (2 << i)
        t += t << offset

    return t


def compile():
    print('Compile time')

    mlp = EvaProgram('NN (MLP)', vec_size=32 * 32)
    with mlp:
        image = Input('input_0')

        d1 = mvp(weights_1, image)
        d1 = d1 + bias_1.tolist()

        act1 = d1 * d1

        d2 = mvp(weights_2, act1)
        d2 = d2 + bias_2.tolist()
        act2 = d2 * d2

        Output('output', act2)
        Output('output', d1)

    mlp.set_output_ranges(60)
    mlp.set_input_scales(60)

    compiler = CKKSCompiler()
    mlp, params, signature = compiler.compile(mlp)

    save(mlp, 'mlp.eva')
    save(params, 'mlp.evaparams')
    save(signature, 'mlp.evasignature')

    # Print IR representation
    with open('mlp.eva', 'rb') as f, open('mlp.txt', 'w') as g:
        read_kt = known_type_pb2.KnownType()
        read_kt.ParseFromString(f.read())
        read_eva = eva_pb2.Program()
        read_eva.ParseFromString(read_kt.contents.value)
        g.write(str(read_eva))


def compute():
    ################################################
    print('Numpy version')
    d1 = np.dot(weights_1, image)
    d1 = d1 + bias_1
    act1 = d1 * d1
    d2 = np.dot(weights_2, act1)
    d2 = d2 + bias_2
    ref_result = d2 * d2
    print(ref_result)

    ################################################
    print('EVA plaintext version')
    mlp = load('mlp.eva')
    eva_ptxt_version = evaluate(mlp, {'input_0': image})
    print(eva_ptxt_version['output'])

    #################################################
    print('Key generation time')

    params = load('mlp.evaparams')

    t0 = time.perf_counter()
    public_ctx, secret_ctx = generate_keys(params)
    t1 = time.perf_counter()
    cur_times['t_keygen'] = delta_ms(t0, t1)

    save(public_ctx, 'mlp.sealpublic')
    save(secret_ctx, 'mlp.sealsecret')

    #################################################
    print('Runtime on client')

    signature = load('mlp.evasignature')
    public_ctx = load('mlp.sealpublic')

    inputs = {
        'input_0': image
    }
    t0 = time.perf_counter()
    encInputs = public_ctx.encrypt(inputs, signature)
    t1 = time.perf_counter()
    cur_times['t_input_encryption'] = delta_ms(t0, t1)

    save(encInputs, 'mlp_inputs.sealvals')

    #################################################
    print('Runtime on server')

    mlp = load('mlp.eva')
    public_ctx = load('mlp.sealpublic')
    encInputs = load('mlp_inputs.sealvals')

    t0 = time.perf_counter()
    encOutputs = public_ctx.execute(mlp, encInputs)
    t1 = time.perf_counter()
    cur_times['t_computation'] = delta_ms(t0, t1)

    save(encOutputs, 'mlp_outputs.sealvals')

    #################################################
    print('Back on client')

    secret_ctx = load('mlp.sealsecret')
    encOutputs = load('mlp_outputs.sealvals')

    t0 = time.perf_counter()
    outputs = secret_ctx.decrypt(encOutputs, signature)
    t1 = time.perf_counter()
    cur_times['t_decryption'] = delta_ms(t0, t1)

    reference = {'output': [ref_result]}
    print('Expected', reference)
    print('Got', outputs)
    print('MSE', valuation_mse(outputs, reference))


def main():
    compile()
    num_runs = int(os.getenv("NUM_RUNS")) if os.getenv("NUM_RUNS") is not None else 10
    for run in range(num_runs):
        global cur_times
        cur_times = copy.copy(times)
        compute()
        print(cur_times)
        all_times.append(cur_times)

        # Output the benchmarking results
        df = pd.DataFrame(all_times)
        output_filename = "mlp_eva.csv"
        if 'OUTPUT_FILENAME' in os.environ:
            output_filename = os.environ['OUTPUT_FILENAME']
        df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    main()
