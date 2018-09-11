package com.konakartadmin.modules.payment;

import com.konakart.app.NameValue;
import com.konakart.app.PaymentOptions;

/**
 * Interface for the Admin Payment Class
 * 
 */
public interface AdminPaymentIf
{
    /**
     * This method executes the transaction with the payment gateway. The action attribute of the
     * options object instructs the method as to what transaction should be executed. E.g. It could
     * be a payment or a payment confirmation for a transaction that has already been authorized
     * etc.
     * 
     * @param options
     * @return Returns an array of NameValue objects that may contain any return information
     *         considered useful by the caller.
     * @throws Exception
     */
    public NameValue[] execute(PaymentOptions options) throws Exception;
}
