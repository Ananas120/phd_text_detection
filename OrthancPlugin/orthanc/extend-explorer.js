$('#instance').live('pagebeforecreate', function() {

  var b = $('<a>')
    .attr('data-role', 'button')
    .attr('href', '#')
    .attr('data-icon', 'search')
    .attr('data-theme', 'e')
    .text('MedTextCleaner');

  b.insertBefore($('#instance-delete').parent().parent());
  b.click(function() {
    if ($.mobile.pageData) {
      var instance = $.mobile.pageData.uuid;

      window.open('/MedTextCleaner/index.html?uuid='+instance);
    }
  });
});
