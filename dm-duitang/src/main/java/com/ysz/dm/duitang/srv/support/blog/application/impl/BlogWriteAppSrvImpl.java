package com.ysz.dm.duitang.srv.support.blog.application.impl;


import com.ysz.dm.duitang.cli.support.blog.dto.AuditBlogReq;
import com.ysz.dm.duitang.cli.support.blog.dto.AuditBlogResp;
import com.ysz.dm.duitang.cli.support.blog.dto.ForwardBlogReq;
import com.ysz.dm.duitang.cli.support.blog.dto.PublishBlogReq;
import com.ysz.dm.duitang.srv.infra.core.AccurateTimeManager;
import com.ysz.dm.duitang.srv.infra.core.IdGenerateManager;
import com.ysz.dm.duitang.srv.support.blog.application.BlogWriteAppSrv;
import com.ysz.dm.duitang.srv.support.blog.domain.adapter.BlogCountAdapter;
import com.ysz.dm.duitang.srv.support.blog.domain.event.BlogEventSender;
import com.ysz.dm.duitang.srv.support.blog.domain.exception.BlogForwardNotForbiddenException;
import com.ysz.dm.duitang.srv.support.blog.domain.exception.BlogIdNotExistException;
import com.ysz.dm.duitang.srv.support.blog.domain.model.Blog;
import com.ysz.dm.duitang.srv.support.blog.domain.model.BlogContentType;
import com.ysz.dm.duitang.srv.support.blog.domain.model.BlogMeta;
import com.ysz.dm.duitang.srv.support.blog.domain.model.BlogText;
import com.ysz.dm.duitang.srv.support.blog.domain.model.BlogUserId;
import com.ysz.dm.duitang.srv.support.blog.domain.repo.BlogRepo;
import java.util.Date;

//@Slf4j
public class BlogWriteAppSrvImpl implements BlogWriteAppSrv {

  private BlogRepo blogRepo;

  private AccurateTimeManager accurateTimeManager;

  private IdGenerateManager idGenerateManager;

  private BlogCountAdapter blogCountAdapter;

  private BlogEventSender blogEventSender;

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

  @Override
  public Blog publishBlog(PublishBlogReq req) {
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
    blogRepo.save(blog);
    blogEventSender.sendBlogPublishedEvent(blog);
    return blog;
  }


  @Override
  public Blog forwardBlog(ForwardBlogReq req) {
    Blog parent = blogRepo.findOne(req.getParentId());
    if (!parent.canBeForward()) {
      throw new BlogForwardNotForbiddenException(req);
    }
    return doForwardBlog(parent, new BlogUserId(req.getUserId()));
  }

  private Blog doForwardBlog(
      final Blog parent,
      final BlogUserId blogUserId) {

    final BlogMeta blogMeta = parent.getBlogMeta();
    final Blog blog = new Blog(
        idGenerateManager.nextBlogId(),
        blogMeta,
        parent.getId(),
        blogUserId,
        accurateTimeManager.now()
    );
    blogRepo.addForward(blog);
    addParentFavCnt(parent); /*面向过程的逻辑*/
    return blog;
  }

  @Override
  public AuditBlogResp blockBlog(AuditBlogReq req) {
    final Long blogId = req.getBlogId();
    final Date now = accurateTimeManager.now();

    final Blog blog = blogRepo.findOne(blogId);
    if (blog == null) {
      /*如果 blog 不存在、意味着数据库有脏数据、 必须抛异常、但是不抛 java 自带异常、而是业务包装异常*/
      throw new BlogIdNotExistException(blogId);
    }

    if (blog.block(req.getCurMsg(), req.getCurStatus(), now)) {
      blogRepo.changeAuditStatus(blog);
      // send blog Blocked event
    }

    // throw block failed Exception ?

    return null;
  }

  private void addParentFavCnt(final Blog parent) {
    blogCountAdapter.incrBlogFavCount(parent.getId(), 1);
    if (!parent.isOrigin()) {
      blogCountAdapter.incrBlogFavCount(parent.rootId(), 1);
    }
  }
}
