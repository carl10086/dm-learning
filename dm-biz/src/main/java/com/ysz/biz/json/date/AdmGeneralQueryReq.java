package com.ysz.biz.json.date;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonFormat.Shape;
import java.io.Serializable;
import java.util.Date;
import lombok.Data;

@Data
public class AdmGeneralQueryReq implements Serializable {

  @JsonFormat(shape = Shape.STRING, pattern = "yyyy-MM-dd HH:mm:ss", timezone = "Asia/Shanghai")
  private Date createAtStart = new Date();
  private Integer start = 0;
  private Integer limit = 24;

  @JsonFormat(shape = Shape.STRING, pattern = "yyyy-MM-dd HH:mm:ss", timezone = "Asia/Shanghai")
  private Date createAtEnd;
  private Long userId = 10L;
  private String status = "22";
}
