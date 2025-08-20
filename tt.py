import pickle
from app import Block, Blockchain, UTXOSet, TXOutput

blockchain = Blockchain()
print("=== Blockchain Status ===")
print(f"Blockchain Height: {len(blockchain.chain)} blocks")
print(f"Total Minted Coins: {blockchain.get_total_minted_coins()} PUK")
for block in blockchain.chain:
    print(f"Block {block.index} | Hash: {block.hash} | TXs: {len(block.transactions)}")
    for tx in block.transactions:
        print(f"  TX {tx.txid}, Outputs: {[(txo.amount, txo.receiver) for txo in tx.outputs]}")

utxo_set = UTXOSet()
print("\n=== UTXO Set ===")
for key, value in utxo_set.db.iterator():
    txo = TXOutput.deserialize(value)
    print(f"UTXO: {key.decode()}, Amount: {txo.amount}, Receiver: {txo.receiver}")