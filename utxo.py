from app import UTXOSet, TXOutput
utxo_set = UTXOSet()
print("=== Current UTXO Set ===")
for key, value in utxo_set.db.iterator():
    txo = TXOutput.deserialize(value)
    print(f"UTXO: {key.decode()}, Amount: {txo.amount}, Receiver: {txo.receiver}")