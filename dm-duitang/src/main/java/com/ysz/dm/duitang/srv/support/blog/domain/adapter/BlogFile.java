package com.ysz.dm.duitang.srv.support.blog.domain.adapter;

import java.util.Date;
import lombok.Data;

@Data
public class BlogFile {

  private final Long id;
  private final String path;
  private final Integer width;
  private final Integer height;
  private final Date createAt;


}
