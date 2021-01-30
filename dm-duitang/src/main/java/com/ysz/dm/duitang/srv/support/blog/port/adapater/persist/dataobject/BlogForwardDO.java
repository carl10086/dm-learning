package com.ysz.dm.duitang.srv.support.blog.port.adapater.persist.dataobject;

import lombok.Data;

@Data
public class BlogForwardDO {

  private Long id;
  private Long rootId;
  private Long parentId;
  // ...

}
