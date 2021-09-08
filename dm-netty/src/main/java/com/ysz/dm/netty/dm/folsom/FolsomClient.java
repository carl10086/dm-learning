package com.ysz.dm.netty.dm.folsom;

import com.spotify.folsom.MemcacheClient;
import com.spotify.folsom.MemcacheClientBuilder;

public class FolsomClient {

  public static void main(String[] args) throws Exception {
    final MemcacheClient<String> client = MemcacheClientBuilder.newStringClient()
        .withAddress("10.1.5.167:11211")
        .connectAscii();



  }

}
