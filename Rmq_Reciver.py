import pika
import json

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='hello')


'''def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)'''
    
def callback(ch, method, properties, body):
    print(" [x] Received %r" % json.loads(body))
    


channel.basic_consume(
    queue='hello', on_message_callback=callback, auto_ack=True)


print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
