//
// (c) 2004-2016 DS Data Systems UK Ltd, All rights reserved.
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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Calendar;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.konakart.app.ExportOrderOptions;
import com.konakart.app.MqOptions;
import com.konakart.app.PdfOptions;
import com.konakart.app.PdfResult;
import com.konakart.appif.ExportOrderOptionsIf;
import com.konakart.appif.ExportOrderResponseIf;
import com.konakart.appif.MqOptionsIf;
import com.konakart.bl.OrderMgr;
import com.konakart.bl.ProductMgr;
import com.konakart.blif.ExportMgrIf;
import com.konakart.db.KKBasePeer;
import com.konakart.db.KKCriteria;
import com.konakart.mqif.MqMgrIf;
import com.konakart.om.BaseOrdersPeer;
import com.konakart.util.KKConstants;
import com.konakartadmin.app.AdminBooking;
import com.konakartadmin.app.AdminCoupon;
import com.konakartadmin.app.AdminCustomer;
import com.konakartadmin.app.AdminOrder;
import com.konakartadmin.app.AdminOrderProduct;
import com.konakartadmin.app.AdminOrderSearch;
import com.konakartadmin.app.AdminOrderSearchResult;
import com.konakartadmin.app.AdminOrderStatusHistory;
import com.konakartadmin.app.AdminOrderUpdate;
import com.konakartadmin.appif.KKAdminIf;
import com.konakartadmin.blif.AdminOrderMgrIf;
import com.konakartadmin.blif.AdminPdfMgrIf;
import com.konakartadmin.blif.AdminPromotionMgrIf;

/**
 * A default implementation of the AdminOrderIntegrationMgrInterface
 */
public class AdminOrderIntegrationMgr extends AdminBaseMgr
        implements AdminOrderIntegrationMgrInterface
{
    /** the log */
    protected static Log log = LogFactory.getLog(AdminOrderIntegrationMgr.class);

    /** Mutex */
    protected static String mutex = "AdminOrderIntegrationMgr";

    /**
     * Constructor
     * 
     * @param eng
     *            KKAdmin engine
     * @throws Exception
     */
    public AdminOrderIntegrationMgr(KKAdminIf eng) throws Exception
    {
        super.init(eng);

        if (log.isDebugEnabled())
        {
            if (eng != null && eng.getEngConf() != null && eng.getEngConf().getStoreId() != null)
            {
                log.debug("AdminOrderIntegrationMgr instantiated for store id = "
                        + eng.getEngConf().getStoreId());
            }
        }
    }

    public void changeOrderStatus(int orderId, int currentStatus, int newStatus)
    {
        if (log.isInfoEnabled())
        {
            log.info("Admin: The order with id = " + orderId
                    + " has just changed state from stateId = " + currentStatus + " to stateId = "
                    + newStatus);
        }

        manageRewardPoints(orderId);
        if (newStatus == OrderMgr.PAYMENT_RECEIVED_STATUS)
        {
            manageDigitalDownloads(orderId);
            manageGiftCertificates(orderId);
            manageGiftRegistries(orderId);
            exportOrder(orderId, KKConstants.UPDATE_OPERATION_TYPE_UPDATE, newStatus);

            // Post the order to the order message queue for processing by some other system
            // postOrderToQueue(orderId);

            // Uncomment to export the order for shipping
            // createOrderExportForShipping(orderId);

            // Uncomment to export a full copy of the order to an XML file
            // createXmlOrderExport(orderId);

            // Uncomment if using TaxCloud
            // manageTaxCloud(orderId);

            // Uncomment if using Bookable Products
            // manageBookings(orderId);

        }

        // By default we don't create any invoices.
        // createInvoice(orderId);

        // Propagate delivery status changes from vendor order to main order
        // manageMultiVendor(orderId, currentStatus, newStatus);
    }

    /**
     * Creates the PDF invoice using the AdminPdfMgr if the code is present.
     * 
     * @param orderId
     */
    protected void createInvoice(int orderId)
    {
        try
        {
            // Create an AdminPdfMgr
            AdminPdfMgrIf pdfMgr = getAdminPdfMgr();

            if (pdfMgr == null)
            {
                // No PDF Manager code present
                return;
            }

            // Create an AdminOrderMgr
            AdminOrderMgrIf orderMgr = getAdminOrderMgr();

            AdminOrder order = orderMgr.getOrderForOrderId(orderId, AdminLanguageMgr.DEFAULT_LANG);

            PdfOptions options = new PdfOptions();
            options.setId(orderId);
            options.setType(KonakartAdminConstants.HTML_ORDER_INVOICE);
            options.setLanguageId(getAdminEng().getLanguageIdForLocale(order.getLocale()));
            options.setReturnFileName(true);
            options.setReturnBytes(false);
            options.setCreateFile(true);

            PdfResult pdfResult = pdfMgr.getPdf(options);

            if (pdfResult != null)
            {
                order.setInvoiceFilename(pdfResult.getFileNameAfterBase());

                // Update the invoice filename on the order
                KKCriteria updateC = getNewCriteria();
                KKCriteria selectC = getNewCriteria();
                updateC.addForInsert(BaseOrdersPeer.INVOICE_FILENAME, order.getInvoiceFilename());
                selectC.add(BaseOrdersPeer.ORDERS_ID, order.getId());
                KKBasePeer.doUpdate(selectC, updateC);
            }
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    /**
     * We get the order from the database and check whether any of the ordered products are digital
     * downloads. For each digital download product we insert a DigitalDownload object in the DB.
     * This is read by the application in order to provide the download link for the customer.
     * 
     * The status of the order is modified if DigitalDownloads exist.
     * 
     * @param orderId
     */
    protected void manageDigitalDownloads(int orderId)
    {
        try
        {
            int numDownloadsCreated = 0;
            AdminProductMgr productMgr = null;
            AdminOrderMgrIf orderMgr = getAdminOrderMgr();

            // Get the order
            AdminOrder order = orderMgr.getOrderForOrderId(orderId, AdminLanguageMgr.DEFAULT_LANG);

            // Check the order products to see whether any of them are digital downloads
            if (order != null && order.getOrderProducts() != null)
            {
                for (int i = 0; i < order.getOrderProducts().length; i++)
                {
                    AdminOrderProduct op = order.getOrderProducts()[i];
                    if (op.getType() == AdminProductMgr.DIGITAL_DOWNLOAD)
                    {
                        if (productMgr == null)
                        {
                            productMgr = new AdminProductMgr(getAdminEng());
                        }
                        productMgr.insertDigitalDownload(order.getCustomerId(), op.getProductId());
                        numDownloadsCreated++;
                    }
                }

                /*
                 * If the order only consisted of digital downloads and we have created the links
                 * for all of the downloads, then we can change the order state to DELIVERED. If it
                 * consisted partially of digital downloads then we change the state to
                 * PARTIALLY_DELIVERED
                 */
                if (numDownloadsCreated == order.getOrderProducts().length)
                {
                    orderMgr.updateOrderStatus(orderId, OrderMgr.DELIVERED_STATUS,
                            "Download Link Available", /* customerNotified */
                            false);
                } else if (numDownloadsCreated > 0)
                {
                    orderMgr.updateOrderStatus(orderId, OrderMgr.PARTIALLY_DELIVERED_STATUS,
                            "Download Link Available", /* customerNotified */
                            false);
                }
            }

        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    /**
     * We get the order from the database and check whether any of the ordered products were ordered
     * from a gift registry. If they were, we have to update the gift registry item (wish list item)
     * quantity received attribute to reflect how many items have been received.
     * 
     * @param orderId
     */
    private void manageGiftRegistries(int orderId)
    {
        try
        {
            String enabled = getAdminConfigMgr()
                    .getConfigurationValue(KonakartAdminConstants.CONF_KEY_ENABLE_GIFT_REGISTRY);
            if (enabled != null && enabled.equalsIgnoreCase("true"))
            {
                AdminOrderMgrIf orderMgr = getAdminOrderMgr();

                // Get the order
                AdminOrder order = orderMgr.getOrderForOrderId(orderId,
                        AdminLanguageMgr.DEFAULT_LANG);

                // Check the order products to see whether any of them are gift registry items
                if (order != null && order.getOrderProducts() != null)
                {
                    for (int i = 0; i < order.getOrderProducts().length; i++)
                    {
                        AdminOrderProduct op = order.getOrderProducts()[i];
                        if (op.getWishListItemId() > 0)
                        {
                            getAdminWishListMgr().updateWishListItemQuantityBought(
                                    op.getWishListItemId(), op.getQuantity());
                        }
                    }
                }
            }
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    /**
     * We get the order from the database and check whether any of the ordered products are gift
     * certificates. If they are we need to process them:
     * <ol>
     * <li>Find the promotion attached to the gift certificate</li>
     * <li>Create a coupon code</li>
     * <li>Create a coupon with the new code</li>
     * <li>Attach the coupon to the promotion</li>
     * <li>Insert the coupon code into a downloadable product that can be downloaded by the
     * customer. This becomes the gift certificate that the customer can forward on to the receiver
     * of the gift.</li>
     * </ol>
     * 
     * @param orderId
     */
    private void manageGiftCertificates(int orderId)
    {
        try
        {
            int numCertificates = 0;
            int numDownloads = 0;
            AdminOrderMgrIf orderMgr = getAdminOrderMgr();
            AdminPromotionMgrIf promMgr = getAdminPromMgr();

            // Get the order
            AdminOrder order = orderMgr.getOrderForOrderId(orderId, AdminLanguageMgr.DEFAULT_LANG);

            // Check the order products to see whether any of them are gift certificate products
            int promotionId = -1;
            if (order != null && order.getOrderProducts() != null)
            {
                for (int i = 0; i < order.getOrderProducts().length; i++)
                {
                    AdminOrderProduct op = order.getOrderProducts()[i];
                    if (op.getType() == AdminProductMgr.GIFT_CERTIFICATE_PRODUCT_TYPE)
                    {
                        numCertificates++;
                        for (int j = 0; j < op.getQuantity(); j++)
                        {
                            promotionId = promMgr.getPromotionIdForGiftCertificate(op);
                            if (promotionId > 0)
                            {
                                // We now create a coupon code
                                String couponCode = getCouponCode(null);

                                // Create a coupon
                                AdminCoupon coupon = new AdminCoupon();
                                coupon.setCouponCode(couponCode);
                                coupon.setMaxUse(1);
                                coupon.setTimesUsed(0);
                                coupon.setName(order.getId() + " - " + order.getCustomerId() + " - "
                                        + order.getCustomerName()); // Set this to whatever
                                // you like

                                // Insert the coupon and Associate it with the promotion
                                promMgr.insertCoupon(coupon, promotionId);

                                /*
                                 * The coupon code could be sent to the customer in whichever way
                                 * the merchant sees fit. In this example we will make it available
                                 * as a digital download product. Alternatively it could be sent as
                                 * an eMail etc.
                                 */
                                String filePath = getGiftCertificateFilePath(op, couponCode);

                                getAdminProdMgr().insertGiftCertificateDigitalDownload(
                                        order.getCustomerId(), op.getProductId(), filePath);

                            } else
                            {
                                log.warn(
                                        "A promotion was not found for Gift Certificate product id = "
                                                + op.getProductId()
                                                + " so it was not processed for order id = "
                                                + op.getOrderId());
                            }
                        }
                    } else if (op.getType() == AdminProductMgr.DIGITAL_DOWNLOAD)
                    {
                        /*
                         * Digital downloads should already have been processed. We keep track of
                         * all gift certificates and digital downloads in order to set the state of
                         * the order.
                         */
                        numDownloads++;
                    }
                }

                /*
                 * If the order only consisted of digital downloads and gift certificates and we
                 * have created the links for all of the downloads, then we can change the order
                 * state to DELIVERED. If it consisted partially of digital downloads and gift
                 * certificates then we change the state to PARTIALLY_DELIVERED
                 */
                if (numCertificates + numDownloads == order.getOrderProducts().length
                        && numCertificates > 0)
                {
                    orderMgr.updateOrderStatus(orderId, OrderMgr.DELIVERED_STATUS,
                            "Download Link Available", /* customerNotified */
                            false);
                } else if (numCertificates > 0)
                {
                    orderMgr.updateOrderStatus(orderId, OrderMgr.PARTIALLY_DELIVERED_STATUS,
                            "Download Link Available", /* customerNotified */
                            false);
                }
            }

        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    /**
     * Creates and returns the path of the file that contains the code of the gift certificate that
     * can be downloaded. This example creates a simple text file. For your store you may want to
     * create a stylish pdf document or an HTML file that the customer can download and send on.
     * 
     * @param op
     * @param couponCode
     * @return Return the file path of the file containing the certificate code
     * @throws IOException
     */
    private String getGiftCertificateFilePath(AdminOrderProduct op, String couponCode)
            throws IOException
    {
        String outputFilename = couponCode + ".txt";
        File myOutFile = new File(outputFilename);

        BufferedWriter bw = new BufferedWriter(new FileWriter(myOutFile));
        bw.write(couponCode);
        bw.close();

        return myOutFile.getAbsolutePath();
    }

    /**
     * Method returns a random coupon code. You may change this code to implement a different
     * algorithm or to call out to an external system.
     * 
     * @param options
     *            Optional parameter to control the type of code returned
     * 
     * @return Returns the coupon code
     */
    public String getCouponCode(String options)
    {
        Random r = new Random();

        char[] possibleChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890".toCharArray();
        int length = 10;

        char[] newCode = new char[length];

        for (int i = 0; i < length; i++)
            newCode[i] = possibleChars[r.nextInt(possibleChars.length)];

        return (new String(newCode));
    }

    /**
     * We get the order from the database, and depending on the state of the order, reward points
     * are assigned to the customer and / or removed from the customer's total. The language for the
     * description of the reward point transaction may be determined by the <code>locale</code>
     * attribute of the order.
     * 
     * @param orderId
     */
    private void manageRewardPoints(int orderId)
    {
        try
        {
            String enabled = getAdminConfigMgr()
                    .getConfigurationValue(KonakartAdminConstants.CONF_KEY_ENABLE_REWARD_POINTS);

            if (enabled != null && enabled.equalsIgnoreCase("true"))
            {
                AdminOrderMgrIf orderMgr = getAdminOrderMgr();

                // Get the order
                AdminOrder order = orderMgr.getOrderForOrderId(orderId,
                        AdminLanguageMgr.DEFAULT_LANG);

                // Order may be null if it belongs to a different store
                if (order == null)
                {
                    return;
                }

                // Check the state
                if (order.getStatus() == OrderMgr.PENDING_STATUS
                        || order.getStatus() == OrderMgr.WAITING_PAYMENT_STATUS)
                {
                    /*
                     * If the customer is redeeming points then we need to reserve those points so
                     * they aren't used again until the order is either paid for or cancelled.
                     */
                    if (order.getPointsRedeemed() > 0 && order.getPointsReservationId() <= 0)
                    {
                        int reservationId = getRewardPointMgr().reservePoints(order.getCustomerId(),
                                order.getPointsRedeemed());
                        // Save the reservation id
                        orderMgr.setRewardPointReservationId(orderId, reservationId);
                    }
                } else if (order.getStatus() == OrderMgr.PAYMENT_RECEIVED_STATUS)
                {
                    /*
                     * We need to commit any reserved points that were reserved when the order was
                     * saved. If the points weren't reserved because the order was saved directly in
                     * the "PAYMENT RECEIVED" state, then we must just delete the points.
                     */
                    if (order.getPointsRedeemed() > 0)
                    {
                        if (order.getPointsReservationId() >= 0)
                        {
                            getRewardPointMgr().deleteReservedPoints(order.getCustomerId(),
                                    order.getPointsReservationId(), "ORDER",
                                    "Points redeemed in order #" + orderId);
                        } else
                        {
                            getRewardPointMgr().deletePoints(order.getCustomerId(),
                                    order.getPointsRedeemed(), "ORDER",
                                    "Points redeemed in order #" + orderId);
                        }
                    }
                    /*
                     * If points were awarded for the order, then they must be assigned to the
                     * customer
                     */
                    if (order.getPointsAwarded() > 0)
                    {
                        getRewardPointMgr().addPoints(order.getCustomerId(),
                                order.getPointsAwarded(), "ORDER",
                                "Points assigned for order #" + orderId);
                    }
                } else if (order.getStatus() == OrderMgr.CANCELLED_STATUS)
                {
                    /*
                     * If the order is cancelled and some points have been reserved, then these
                     * points must be returned to the customer.
                     */
                    if (order.getPointsRedeemed() > 0 && order.getPointsReservationId() >= 0)
                    {
                        getRewardPointMgr().freeReservedPoints(order.getCustomerId(),
                                order.getPointsReservationId());
                    }
                }
            }
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    /**
     * Creates an Order Export file for the order.
     * <p>
     * Typically this would be used to create an import file for the purposes of integrating
     * KonaKart with a 3rd Party system such as UPS WorldShip
     * 
     * @param orderId
     * @return a ExportOrderResponse object which can be null if the export was unsuccessful
     */
    @SuppressWarnings("unused")
    private ExportOrderResponseIf createOrderExportForShipping(int orderId)
    {
        try
        {
            ExportOrderOptionsIf options = new ExportOrderOptions();
            options.setOrderId(orderId);
            options.setCode(KKConstants.EXP_ORDER_BY_SHIPPING_MODULE);

            ExportOrderResponseIf response = getAdminOrderMgr()
                    .exportOrder((ExportOrderOptions) options);

            if (response != null)
            {
                if (log.isDebugEnabled())
                {
                    log.debug(response.toString());
                }
            } else
            {
                if (log.isDebugEnabled())
                {
                    log.debug("exportOrder returned = null");
                }
            }

            return response;
        } catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Creates an XML Order Export file for the order.
     * 
     * @param orderId
     * @return a ExportOrderResponse object which can be null if the export was unsuccessful
     */
    @SuppressWarnings("unused")
    private ExportOrderResponseIf createXmlOrderExport(int orderId)
    {
        try
        {
            ExportOrderOptionsIf options = new ExportOrderOptions();
            options.setOrderId(orderId);
            options.setCode(KKConstants.EXP_ORDER_FULL_ORDER_TO_XML);

            ExportOrderResponseIf response = getAdminOrderMgr()
                    .exportOrder((ExportOrderOptions) options);

            if (response != null)
            {
                if (log.isDebugEnabled())
                {
                    log.debug(response.toString());
                }
            } else
            {
                if (log.isDebugEnabled())
                {
                    log.debug("exportOrder returned = null");
                }
            }

            return response;
        } catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Exports the order for ERP XML Integration.
     * 
     * @param orderId
     *            the id of the order to post to the Message Queue
     * @param operationType
     *            The type of operation
     * @param newStatus
     *            The new order state
     */
    private void exportOrder(int orderId, int operationType, int newStatus)
    {
        try
        {
            if (isEnterprise())
            {
                ExportMgrIf exportMgr = getExportMgr();
                if (exportMgr.isERPEnabled())
                {
                    if (exportMgr.isExportOrder())
                    {
                        exportMgr.exportOrder(getStoreId(), orderId);
                    }

                    if (exportMgr.isUpdateObjectEnabled(KKConstants.UPDATE_OBJECT_TYPE_ORDER,
                            operationType))
                    {
                        exportMgr.updateObject(getStoreId(), KKConstants.UPDATE_OBJECT_TYPE_ORDER,
                                operationType, orderId, /* objectId1 */null, /* objectId2 */null,
                                /* value1 */Integer.toString(newStatus), /* value2 */null);
                    }
                }
            }
        } catch (Throwable e)
        {
            log.error("Problem exporting order to queue", e);
        }
    }

    /**
     * Posts the order to the Order Message Queue
     * 
     * @param orderId
     *            the id of the order to post to the Message Queue
     */
    @SuppressWarnings("unused")
    private void postOrderToQueue(int orderId)
    {
        try
        {
            ExportOrderOptionsIf exportOptions = new ExportOrderOptions();
            exportOptions.setOrderId(orderId);
            exportOptions.setCode(KKConstants.EXP_ORDER_RETURN_XML_STRING);

            ExportOrderResponseIf response = getAdminOrderMgr()
                    .exportOrder((ExportOrderOptions) exportOptions);

            if (response == null)
            {
                if (log.isWarnEnabled())
                {
                    log.warn("exportOrder returned = null");
                }
                return;
            }

            if (log.isDebugEnabled())
            {
                log.debug(response.toString());
            }

            MqOptionsIf mqOptions = new MqOptions();
            mqOptions.setQueueName(getPropertyValue("konakart.mq.orders.queue"));
            mqOptions.setBrokerUrl(getPropertyValue("konakart.mq.broker.uri"));
            mqOptions.setUsername(getPropertyValue("konakart.mq.username"));
            mqOptions.setPassword(getPropertyValue("konakart.mq.password"));

            mqOptions.setMsgText(response.getOrderAsXml());
            MqMgrIf mgr = getMqMgr();
            mgr.postMessageToQueue(mqOptions);

            if (log.isInfoEnabled())
            {
                log.info("Order " + orderId + " Posted Successfully to the Message Queue");
            }
        } catch (Exception e)
        {
            log.warn("Problem posting Order to Message Queue");
            e.printStackTrace();
        }
    }

    /**
     * The order details are sent to the tax cloud tax service.
     * 
     * @param orderId
     */
    @SuppressWarnings("unused")
    private void manageTaxCloud(int orderId)
    {
        if (log.isDebugEnabled())
        {
            log.debug("Entered manageTaxCloud()");
        }
        try
        {
            String enabled = getAdminConfigMgr()
                    .getConfigurationValue("MODULE_ORDER_TOTAL_TAX_CLOUD_STATUS");

            if (enabled != null && enabled.equalsIgnoreCase("true"))
            {
                // Get the order
                AdminOrder order = getAdminOrderMgr().getOrderForOrderId(orderId,
                        AdminLanguageMgr.DEFAULT_LANG);

                // Call TaxCloud to authorize and capture the transaction
                net.taxcloud.service.AuthorizedService authorizedService = new net.taxcloud.service.AuthorizedService();
                String apiLoginId = getAdminConfigMgr()
                        .getConfigurationValue("MODULE_ORDER_TOTAL_TAX_CLOUD_API_LOGIN_ID");
                String apiKey = getAdminConfigMgr()
                        .getConfigurationValue("MODULE_ORDER_TOTAL_TAX_CLOUD_API_LOGIN_KEY");
                String customerId = String.valueOf(order.getCustomerId());
                String cartId = order.getLifecycleId(); // Unique ID for cart generated when order
                // was created
                String orderIdStr = String.valueOf(order.getId()); // Unique ID for order
                Calendar dateAuthorized = Calendar.getInstance();
                boolean authorized = authorizedService.authorized(apiLoginId, apiKey, customerId,
                        cartId, orderIdStr, dateAuthorized);
                if (log.isDebugEnabled())
                {
                    log.debug("Authorized returned: " + authorized + " for\n customerId = "
                            + customerId + "\n cartId = " + cartId + "\n orderId = " + orderIdStr);
                }

                if (authorized)
                {
                    net.taxcloud.service.CapturedService capturedService = new net.taxcloud.service.CapturedService();
                    boolean captured = capturedService.captured(apiLoginId, apiKey, orderIdStr);
                    log.debug("Captured returned: " + captured + " for orderId = " + orderIdStr);
                }
            }

        } catch (Exception e)
        {
            log.warn("Problem authorizing / captuing TaxCloud transaction");
            e.printStackTrace();
        }
    }

    /**
     * We get the order from the database and check whether any of the ordered products are bookable
     * products. For each bookable product we insert a Booking object in the DB. At the moment we
     * assume that the name of the person in the Booking object is the name of the customer. For the
     * case where a customer may make multiple bookings providing the names of the people, these
     * names will have to be encoded into a custom field of the order product and read from the
     * custom field in this method, so that a Booking object may be inserted for each person.
     * 
     * @param orderId
     */
    @SuppressWarnings("unused")
    private void manageBookings(int orderId)
    {
        if (log.isDebugEnabled())
        {
            log.debug("Entered manageBookings()");
        }
        AdminOrder order = null;
        try
        {
            // Get the order
            order = getAdminOrderMgr().getOrderForOrderId(orderId, AdminLanguageMgr.DEFAULT_LANG);

            // Check the order products to see whether any of them are bookable products
            if (order != null && order.getOrderProducts() != null)
            {
                for (int i = 0; i < order.getOrderProducts().length; i++)
                {
                    AdminOrderProduct op = order.getOrderProducts()[i];
                    if (op.getType() == ProductMgr.BOOKABLE_PRODUCT_TYPE)
                    {
                        AdminBooking booking = new AdminBooking();
                        booking.setCustomerId(order.getCustomerId());
                        booking.setOrderId(order.getId());
                        booking.setOrderProductId(op.getId());
                        booking.setQuantity(op.getQuantity());
                        booking.setProductId(op.getProductId());
                        AdminCustomer customer = getAdminCustMgr()
                                .getCustomerForId(order.getCustomerId());
                        if (customer != null)
                        {
                            booking.setFirstName(customer.getFirstName());
                            booking.setLastName(customer.getLastName());
                        }
                        getAdminBookableProductMgr().insertBooking(booking, null);
                    }
                }
            }

        } catch (Exception e)
        {
            if (order == null)
            {
                log.warn("Problem creating bookings");
            } else
            {
                log.warn("Problem creating bookings for order id = " + order.getId());
            }
            e.printStackTrace();
        }
    }

    /**
     * We determine whether this order has a parent. If it does, we decide whether to add the status
     * change to the parent. This method should be customized to include your business process.<br>
     * This example implementation defines a couple of states (Delivered and Partially Delivered)
     * and adds them to the status trail of the main order. If the <code>newStatus</code> is
     * Delivered then we look at the current status of all of the vendor orders to see whether they
     * have all been fully delivered. If this is the case then we set the status of the parent order
     * to delivered otherwise to partially delivered.
     * 
     * @param orderId
     */
    @SuppressWarnings("unused")
    private void manageMultiVendor(int orderId, int currentStatus, int newStatus)
    {
        if (log.isDebugEnabled())
        {
            log.debug("Entered manageMultiVendor()");
        }

        AdminOrder order = null;
        try
        {
            // Get the order
            order = getAdminOrderMgr().getOrderForOrderId(orderId, AdminLanguageMgr.DEFAULT_LANG);
            if (order == null)
            {
                return;
            }

            int parentId = order.getParentId();
            if (parentId <= 0)
            {
                return;
            }

            AdminOrderMgrIf mgr = getAdminOrderMgr();

            /*
             * If the new state of the vendor order is DELIVERED then we must determine whether all
             * of the other vendor orders are in the DELIVERED state. If this is the case then we
             * can set the state of the parent order to DELIVERED otherwise we set it to
             * PARTIALLY_DELIVERED
             */
            boolean allDelivered = true;
            if (newStatus == OrderMgr.DELIVERED_STATUS)
            {
                // Get all vendor orders for the parent
                AdminOrderSearch search = new AdminOrderSearch();
                search.setParentId(parentId);
                AdminOrderSearchResult result = mgr.getOrders(search, 0, 100,
                        AdminLanguageMgr.DEFAULT_LANG);
                AdminOrder[] vendorOrders = null;
                if (result != null)
                {
                    vendorOrders = result.getOrders();
                }

                /*
                 * Figure out from the vendor orders whether they have all been delivered.
                 */
                if (vendorOrders != null)
                {
                    for (int i = 0; i < vendorOrders.length; i++)
                    {
                        AdminOrder vendorOrder = vendorOrders[i];
                        if (vendorOrder.getStatus() != OrderMgr.DELIVERED_STATUS)
                        {
                            allDelivered = false;
                            break;
                        }
                    }
                }
            } else
            {
                allDelivered = false;
            }

            AdminOrderStatusHistory[] statusTrail = order.getStatusTrail();
            AdminOrderStatusHistory currentStatusObject = null;
            if (statusTrail != null && statusTrail.length > 0)
            {
                currentStatusObject = statusTrail[statusTrail.length - 1];
                if (currentStatusObject.getOrderStatusId() == OrderMgr.PARTIALLY_DELIVERED_STATUS
                        || currentStatusObject.getOrderStatusId() == OrderMgr.DELIVERED_STATUS)
                {
                    /*
                     * We update the status of the parent order. The customer may have already been
                     * notified during the status change of the vendor order so we set that to
                     * false.
                     */
                    // Get the parent order
                    AdminOrder parentOrder = mgr.getOrderForOrderId(parentId,
                            AdminLanguageMgr.DEFAULT_LANG, /* allStores */true);
                    if (parentOrder == null)
                    {
                        throw new Exception("Cannot find parent order for order id = " + parentId
                                + ". Id of child is " + orderId);
                    }

                    int newState = (allDelivered) ? OrderMgr.DELIVERED_STATUS
                            : OrderMgr.PARTIALLY_DELIVERED_STATUS;
                    AdminOrderUpdate update = new AdminOrderUpdate();
                    update.setUpdatedById(currentStatusObject.getUpdatedById());
                    mgr.updateOrder(parentId, newState,
                            currentStatusObject.getComments(), /* notifyCustomer */
                            false, update, parentOrder.getStoreId());
                }
            }

        } catch (Exception e)
        {
            if (order == null)
            {
                log.error("Problem managing multi-vendor orders", e);
            } else
            {
                log.error("Problem managing multi-vendor orders for order id = " + order.getId(),
                        e);
            }
        }
    }

    /**
     * This method may be customized in order to implement an algorithm that creates an RMA code for
     * the order. The Administration Application will use the returned value (if not null) to
     * automatically populate the RMA code entry field.
     * 
     * @param orderId
     *            Id of the order
     * @return Returns an RMA Code
     */
    public String getRMACode(int orderId)
    {
        return orderId + "-" + System.currentTimeMillis() / 1000;
    }
}
