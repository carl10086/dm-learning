package com.ysz.dm.duitang.cli.support.blog.dto;

import java.io.Serializable;
import lombok.Data;

@Data
public class DetailQueryCond implements Serializable {

  private boolean fillUser;
  private boolean fillFile;
  private boolean fillCount;

}
