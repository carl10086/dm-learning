package com.ysz.dm.duitang.srv.support.blog.domain.factory;

import com.ysz.dm.duitang.cli.support.blog.dto.PublishBlogReq;
import com.ysz.dm.duitang.srv.infra.core.AccurateTimeManager;
import com.ysz.dm.duitang.srv.infra.core.IdGenerateManager;
import com.ysz.dm.duitang.srv.support.blog.domain.model.Blog;
import com.ysz.dm.duitang.srv.support.blog.domain.model.BlogContentType;
import com.ysz.dm.duitang.srv.support.blog.domain.model.BlogMeta;
import com.ysz.dm.duitang.srv.support.blog.domain.model.BlogText;
import com.ysz.dm.duitang.srv.support.blog.domain.model.BlogUserId;
import java.util.Date;

public class BlogFactory {


  private IdGenerateManager idGenerateManager;

  private AccurateTimeManager accurateTimeManager;

  private BlogMeta newBlogMeta(PublishBlogReq req) {
    final Date now = accurateTimeManager.now();
    return new BlogMeta(
        idGenerateManager.nextBlogId(),
        new BlogText(req.getMsg()),
        null /*extra 假装已经有了*/,
        now,
        BlogContentType.valueOf(req.getContentType()),
        new BlogUserId(req.getUserId()),
        req.getPhotoId()
    );
  }

  public Blog createOriginBlog(PublishBlogReq req) {
    /*原发发布的时候、包含了 BlogMeta 对象的创建逻辑, 是否应该抽象为一个 Factory , factory 应该是无状态的 . */
    final BlogMeta blogMeta = newBlogMeta(req);

    /*在生成一个原发(BlogMeta) 的时候、也要生成一个默认的转发关系作为根(BlogForward)*/
    Blog blog = new Blog(
        blogMeta.getId(),
        blogMeta,
        null,
        blogMeta.getBlogUserId(),
        blogMeta.getCreateAt()
    );

    return blog;
  }
}
