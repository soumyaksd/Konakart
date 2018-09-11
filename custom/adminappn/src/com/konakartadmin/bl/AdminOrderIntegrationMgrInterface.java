//
// (c) 2004-2015 DS Data Systems UK Ltd, All rights reserved.
//
// DS Data Systems and KonaKart and their respective logos, are 
// trademarks of DS Data Systems UK Ltd. All rights reserved.
//
// The information in this document is the proprietary property of
// DS Data Systems UK Ltd. and is protected by English copyright law,
// the laws of foreign jurisdictions, and international treaties,
// as applicable. No part of this document may be reproduced,
// transmitted, transcribed, transferred, modified, published, or
// translated into any language, in any form or by any means, for
// any purpose other than expressly permitted by DS Data Systems UK Ltd.
// in writing.
//
package com.konakartadmin.bl;

/**
 * Defines the interface for the AdminOrderIntegrationMgr
 */
public interface AdminOrderIntegrationMgrInterface
{
    /**
     * Called just after an order status change
     * 
     * @param orderId
     *            Id of the order
     * @param currentStatus
     *            Current state of the order
     * @param newStatus
     *            New state of the order
     */
    public void changeOrderStatus(int orderId, int currentStatus, int newStatus);

    /**
     * This method may be customized in order to implement an algorithm that creates an RMA code for
     * the order. The Administration Application will use the returned value (if not null) to
     * automatically populate the RMA code entry field.
     * 
     * @param orderId
     *            Id of the order
     * @return Returns an RMA Code
     */
    public String getRMACode(int orderId);

    /**
     * This method may be customized in order to implement an algorithm that creates a coupon code
     * or to call out to an external system in order to retrieve one.
     * 
     * @param options
     *            Optional parameter to control the type of code returned
     * 
     * @return Returns the coupon code
     */
    public String getCouponCode(String options);
}
