package com.ysz.dm.fast.socket.nio;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.util.Iterator;
import java.util.Objects;
import java.util.Set;

/**
 * @author carl
 */
public class EchoServer {


  private int port = 1234;
  private int bufSize = 1024;

  private volatile boolean close = false;


  public void start() {
    ServerSocketChannel server;
    try {
      server = ServerSocketChannel.open();
      server.socket().bind(new InetSocketAddress(port));
      server.socket().setReuseAddress(true);
      server.configureBlocking(false);

      Selector selector = Selector.open();

      server.register(selector, SelectionKey.OP_ACCEPT);

      ByteBuffer buffer = ByteBuffer.allocate(bufSize);

      while (!close) {
        int cnt = selector.select();

        if (cnt > 0) {
          Set<SelectionKey> selectionKeys = selector.selectedKeys();
          Iterator<SelectionKey> iterator = selectionKeys.iterator();
          while (iterator.hasNext()) {
            SelectionKey next = iterator.next();
            iterator.remove();

            if (next.isAcceptable()) {
              SocketChannel clientChannel = server.accept();
              clientChannel.configureBlocking(false);
              clientChannel
                  .register(selector, SelectionKey.OP_READ, clientChannel.socket().getPort());
            } else if (next.isReadable()) {
              SocketChannel clientChannel = (SocketChannel) next.channel();
              System.out.println("port:" + Objects.toString(next.attachment()));

              if (clientChannel.read(buffer) < 0) {
                /*1. <0 证明读完了*/
                next.cancel();
              } else {
                buffer.flip();
                clientChannel.write(buffer);
                buffer.clear();
              }
            }
          }
        }
      }
      server.close();
    } catch (IOException e) {
      e.printStackTrace();
      throw new RuntimeException();
    }


  }


  public static void main(String[] args) {
    new EchoServer().start();
  }

}
