package com.ysz.dm.duitang.cli.support.blog.dto;

import java.io.Serializable;
import java.util.Date;
import lombok.Data;

@Data
public class BlogEventDTO implements Serializable {

  private Long blogId;
  private String eventType;
  private Date occurAt;

}
