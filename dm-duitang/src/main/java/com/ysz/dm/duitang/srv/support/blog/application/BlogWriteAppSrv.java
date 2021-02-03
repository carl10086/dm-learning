package com.ysz.dm.duitang.srv.support.blog.application;

import com.ysz.dm.duitang.cli.support.blog.dto.AuditBlogReq;
import com.ysz.dm.duitang.cli.support.blog.dto.AuditBlogResp;
import com.ysz.dm.duitang.cli.support.blog.dto.ForwardBlogReq;
import com.ysz.dm.duitang.cli.support.blog.dto.PublishBlogReq;
import com.ysz.dm.duitang.srv.support.blog.domain.model.Blog;

public interface BlogWriteAppSrv {

  /**
   * 发布原发 blog
   */
  Blog publishBlog(PublishBlogReq req);


  /**
   * 转发 blog
   */
  Blog forwardBlog(ForwardBlogReq req);


  AuditBlogResp blockBlog(AuditBlogReq req);
}
