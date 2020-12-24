# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from eva import Program, EvaProgram, Input, Output, evaluate, save, load, Type, Op, Term
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse
import numpy as np
from google.protobuf.json_format import MessageToJson
from google.protobuf import text_format
from google.protobuf.any_pb2 import Any
import known_type_pb2
import eva_pb2
import vec_lang_pb2


def decode_eva(eva_file, text_file):
    with open(eva_file, 'rb') as f, open(text_file, 'w') as g:
        read_kt = known_type_pb2.KnownType()
        read_kt.ParseFromString(f.read())
        print(str(read_kt))
        read_eva = eva_pb2.Program()
        read_eva.ParseFromString(read_kt.contents.value)
        g.write(str(read_eva))


def compile():
    #################################################
    # EVA
    print('Compile time')

    poly = EvaProgram('Polynomial', vec_size=1)
    with poly:
        x = Input('x')
        Output('y', 3 * x)

    poly.set_output_ranges(66)
    poly.set_input_scales(77)

    compiler = CKKSCompiler()
    poly_compiled, params, signature = compiler.compile(poly)

    save(poly_compiled, 'poly_compiled.eva')
    save(poly, 'poly_nocompile.eva')
    save(params, 'poly.evaparams')
    save(signature, 'poly.evasignature')

    #################################################
    # MANUAL
    poly_manual = Program(name='Poly Manual', vec_size=1)
    input_term = poly_manual._make_input('x', Type.Cipher)
    const_term = poly_manual._make_uniform_constant(3.0)
    # There does not seem to be a way to set attributes through the python interface
    # In addition, it seems like encode & rescale operations are inserted by the compiler
    product = poly_manual._make_term(Op.Mul, [const_term, input_term])
    poly_manual._make_output('y', product)
    poly_manual.set_output_ranges(66)
    poly_manual.set_input_scales(77)
    compiler = CKKSCompiler()
    poly_manual_compiled, params_manual, signature_manual = compiler.compile(poly_manual)
    save(poly_manual, 'poly_manual_nocompile.eva')
    save(poly_manual_compiled, 'poly_manual_compiled.eva')


    #################################################
    # DECODE ALL
    decode_eva('poly_manual_nocompile.eva', 'poly_manual_nocompile.txt')
    decode_eva('poly_manual_compiled.eva', 'poly_manual_compiled.txt')
    decode_eva('poly_nocompile.eva', 'poly_nocompile.txt')
    decode_eva('poly_compiled.eva', 'poly_compiled.txt')
    
    with open('small_mnist_hw.vec', 'rb') as f, open('small_mnist_hw.txt', 'w') as g:
        read_vl = vec_lang_pb2.Program()
        read_vl.ParseFromString(f.read())
        g.write(str(read_vl))


def run_program():
    #################################################
    print('Key generation time')

    params = load('poly.evaparams')

    public_ctx, secret_ctx = generate_keys(params)

    save(public_ctx, 'poly.sealpublic')
    save(secret_ctx, 'poly.sealsecret')

    #################################################
    print('Runtime on client')

    signature = load('poly.evasignature')
    public_ctx = load('poly.sealpublic')

    inputs = {
        'x': [4]
    }
    encInputs = public_ctx.encrypt(inputs, signature)

    save(encInputs, 'poly_inputs.sealvals')

    #################################################
    print('Runtime on server')

    poly = load('poly_manual_compiled.eva')
    public_ctx = load('poly.sealpublic')
    encInputs = load('poly_inputs.sealvals')

    encOutputs = public_ctx.execute(poly, encInputs)

    save(encOutputs, 'poly_outputs.sealvals')

    #################################################
    print('Back on client')

    poly_ref = load('poly_compiled.eva')
    secret_ctx = load('poly.sealsecret')
    encOutputs = load('poly_outputs.sealvals')

    outputs = secret_ctx.decrypt(encOutputs, signature)

    reference = evaluate(poly_ref, inputs)
    print('Expected', reference)
    print('Got', outputs)
    print('MSE', valuation_mse(outputs, reference))


def main():
    compile()
    run_program()


if __name__ == "__main__":
    main()
