package com.ysz.biz.json.date;

import com.fasterxml.jackson.databind.ObjectMapper;

public class FastJson_Dm_001 {


  public static void main(String[] args) throws Exception {
    AdmGeneralQueryReq req = new AdmGeneralQueryReq();
    ObjectMapper mapper = new ObjectMapper();
    final String value = mapper.writeValueAsString(req);

    System.err.println(value);
    System.err.println(mapper.readValue(value, AdmGeneralQueryReq.class));
  }
}
